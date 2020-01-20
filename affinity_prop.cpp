#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <map>
#include <exception>
#include <algorithm>
#include <functional>


typedef unsigned int ui32;

struct TEdge {
    ui32 Node1;
    ui32 Node2;

    bool operator<(const TEdge& rhs) const {
        return std::tie(Node1, Node2) < std::tie(rhs.Node1, rhs.Node2);
    }

    bool operator>(const TEdge& rhs) const {
        return std::tie(Node1, Node2) > std::tie(rhs.Node1, rhs.Node2);
    }

    bool operator>=(const TEdge& rhs) const {
        return !(*this < rhs);
    }

    bool operator<=(const TEdge& rhs) const {
        return !(*this > rhs);
    }

    bool operator==(const TEdge& rhs) const {
        return Node1 == rhs.Node1 && Node2 == rhs.Node2;
    }

    bool operator!=(const TEdge& rhs) const {
        return !(*this == rhs);
    }
};

namespace std {
    template <>
    struct hash<TEdge>{
        std::size_t operator()(const TEdge& edge) const {
            std::hash<ui32> hasher{};
            return hasher(edge.Node1) ^ hasher(edge.Node2);
        }
    };
}

class TEdgeFunction {
public:
    TEdgeFunction() {}
    // TEdgeFunction(const std::unordered_map<TEdge, float>& values)
    //     : Values(values)
    // {
    //     for (auto& [edge, value]: values) {
    //         Edges_.push_back(edge);
    //     }
    // }

    const std::set<TEdge>& Edges() const {
        return Edges_;
    }

    float& operator[](const TEdge& edge) {
        if (Edges_.find(edge) == Edges_.end()) {
            Edges_.insert(edge);
            Values[edge] = 0.;
        }

        return Values[edge];
    }

    float operator[](const TEdge& edge) const {
        return Values.at(edge);
    }

private:
    std::set<TEdge> Edges_;
    std::unordered_map<TEdge, float> Values;
};


class TGraph {
public:
    TGraph() {}

    void AddEdge(ui32 node1, ui32 node2) {
        AdjacencyList[node1].insert(node2);
        AdjacencyList[node2].insert(node1);
        AdjacencyList[node1].insert(node1);
        AdjacencyList[node2].insert(node2);
        Vertexes_.insert(node1);
        Vertexes_.insert(node2);
    }

    const std::set<ui32>& Vertexes() const {
        return Vertexes_;
    }

    const std::set<ui32>& GetAdjacents(ui32 nodeIdx) const {
        return AdjacencyList.at(nodeIdx);
    }

    TEdgeFunction CreateEmptyEdgeFunction(float fillValue) const {
        TEdgeFunction edgeFunction;
        for (const auto& [node1, adjacents] : AdjacencyList) {
            for (ui32 node2 : adjacents) {
                TEdge edge{node1, node2};
                edgeFunction[edge] = fillValue;
            }
        }
        return edgeFunction;
    }

private:
    std::set<ui32> Vertexes_;
    std::unordered_map<ui32, std::set<ui32>> AdjacencyList;
};

void UpdateAvailibility(
    const TGraph& graph,
    const TEdgeFunction& similarity,
    const TEdgeFunction& responsibility,
    TEdgeFunction& availibility
) {
    std::unordered_map<ui32, float> sumResponses;
    for (ui32 nodeIdx: graph.Vertexes()) {
        sumResponses[nodeIdx] = responsibility[TEdge{nodeIdx, nodeIdx}];
    }

    for (ui32 nodeIdx: graph.Vertexes()) {
        for (ui32 adjacentNode: graph.GetAdjacents(nodeIdx)) {
            TEdge edge{nodeIdx, adjacentNode};
            sumResponses[adjacentNode] += std::max(0.f, responsibility[edge]);
        }
    }

    for (const auto& edge: availibility.Edges()) {
        const float value = sumResponses[edge.Node2] - std::max(0.f, responsibility[edge]);

        if (edge.Node1 == edge.Node2) {
            availibility[edge] = value - responsibility[edge];
        } else {
            availibility[edge] = std::min(0.f, value);
        }
    }
}

TEdge FindMaxNeighbour(
    ui32 nodeIdx,
    const std::set<ui32> adjacents,
    const TEdgeFunction& similarity,
    const TEdgeFunction& availibility,
    TEdge* bestIsMe = nullptr
) {
    TEdge maxEdge{nodeIdx, *adjacents.begin()};
    if (bestIsMe != nullptr && maxEdge == *bestIsMe) {
        maxEdge = TEdge{nodeIdx, *(++adjacents.begin())};
    }
    float maxValue = similarity[maxEdge] + availibility[maxEdge];

    for (ui32 adjacentNode : adjacents) {
        TEdge edge{nodeIdx, adjacentNode};
        const float value = similarity[edge] + availibility[edge];
        if (value > maxValue) {
            if (bestIsMe != nullptr && edge == *bestIsMe) {
                continue;
            }
            maxEdge = edge;
            maxValue = value;
        }
    }

    return maxEdge;
}

void UpdateResponsibility(
    const TGraph& graph,
    const TEdgeFunction& similarity,
    const TEdgeFunction& availibility,
    TEdgeFunction& responsibility
) {
    std::unordered_map<ui32, TEdge> bestNeighbour;

    for (ui32 nodeIdx : graph.Vertexes()) {
        const auto maxEdge = FindMaxNeighbour(
            nodeIdx,
            graph.GetAdjacents(nodeIdx),
            similarity,
            availibility
        );

        bestNeighbour[nodeIdx] = maxEdge;
    }

    for (const auto& edge: responsibility.Edges()) {
        TEdge bestEdge = bestNeighbour[edge.Node1];

        if (bestEdge == edge) {
            const ui32 nodeIdx = edge.Node1;
            const auto maxEdge = FindMaxNeighbour(
                nodeIdx,
                graph.GetAdjacents(nodeIdx),
                similarity,
                availibility,
                &bestEdge
            );
            assert(bestEdge != maxEdge);
            bestEdge = maxEdge;
        }

        const float value = availibility[bestEdge] + similarity[bestEdge];
        responsibility[edge] = similarity[edge] - value;
    }
}

void PrintEdgeFunction(const TGraph& graph, const TEdgeFunction& edgeFunction) {
    using namespace std;
    for (ui32 nodeIdx: graph.Vertexes()) {
        for (ui32 adjacentNode: graph.GetAdjacents(nodeIdx)) {
            cout << "(" << nodeIdx << "," << adjacentNode << ") = "
                << edgeFunction[TEdge{nodeIdx, adjacentNode}] << endl;
        }
    }
}


void AffinityPropagation(
    const TGraph& graph,
    std::map<ui32, ui32>* nodeToClusterIndex
) {
    ui32 nIterations = 100;

    TEdgeFunction similarity = graph.CreateEmptyEdgeFunction(-1.f);

    for (ui32 nodeIdx: graph.Vertexes()) {
        TEdge selfEdge{nodeIdx, nodeIdx};
        similarity[selfEdge] = -10000.f;
    }
    // std::cout << "sim" << std::endl;
    // PrintEdgeFunction(graph, similarity);

    TEdgeFunction availibility = graph.CreateEmptyEdgeFunction(0.f);
    TEdgeFunction responsibility = graph.CreateEmptyEdgeFunction(0.f);

    for (ui32 iter = 0; iter < nIterations; iter++) {
        std::cout << "iter " << iter << std::endl;
        UpdateResponsibility(graph, similarity, availibility, responsibility);
        UpdateAvailibility(graph, similarity, responsibility, availibility);

        // std::cout << "resp" << std::endl;
        // PrintEdgeFunction(graph, responsibility);
        // std::cout << "av" << std::endl;
        // PrintEdgeFunction(graph, availibility);
    }

    std::unordered_map<ui32, ui32> clusterToIdx;

    for (ui32 nodeIdx: graph.Vertexes()) {
        const auto& adjacents = graph.GetAdjacents(nodeIdx);

        ui32 clusterIdx = 0;
        float maxValue = 0;
        {
            TEdge firstEdge{nodeIdx, *adjacents.begin()};
            maxValue = availibility[firstEdge] + responsibility[firstEdge];
        }
        for (ui32 adjacentNode: adjacents) {
            TEdge edge{nodeIdx, adjacentNode};
            const float value = availibility[edge] + responsibility[edge];
            if (value > maxValue) {
                maxValue = value;
                clusterIdx = adjacentNode;
            }
        }

        if (clusterToIdx.find(clusterIdx) == clusterToIdx.end()) {
            clusterToIdx[clusterIdx] = clusterToIdx.size();
        }
        nodeToClusterIndex->insert(std::make_pair(nodeIdx, clusterToIdx[clusterIdx]));
    }
}


int main() {
    using namespace std;

    ifstream inputFile("input.txt");

    TGraph graph;

    string line;
    while (getline(inputFile, line)) {
        stringstream ss(line);
        ui32 node1;
        ui32 node2;
        ss >> node1 >> node2;

        graph.AddEdge(node1, node2);
    }

    std::map<ui32, ui32> nodeToClusterIndex;
    AffinityPropagation(graph, &nodeToClusterIndex);

    for (auto& [nodeIdx, clusterIdx]: nodeToClusterIndex) {
        cout << "node, cluster " << nodeIdx << " " << clusterIdx << endl;
    }

    return 0;
}
