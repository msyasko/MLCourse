#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <exception>
#include <algorithm>
#include <functional>
#include <cstdlib>


typedef unsigned int ui32;
typedef ui32 TLocationId;

//std::unordered_map<ui32, std::vector<ui32>> ovResult;

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
            return hasher(edge.Node1 * 1000000 + edge.Node2);
        }
    };
}

class TEdgeFunction {
public:
    TEdgeFunction() {
        Edges_.reserve(10 * 1000000);
        Values.reserve(10 * 1000000);
    }

    const std::unordered_set<TEdge>& Edges() const {
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
    std::unordered_set<TEdge> Edges_;
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


void Noise(TEdgeFunction& edgeFunction) {
    std::srand(unsigned(std::time(0)));
    for (auto& edge: edgeFunction.Edges()) {
        edgeFunction[edge] += (static_cast<float>(std::rand()) / RAND_MAX) * 1e-1;
    }
}


void AffinityPropagation(
    const TGraph& graph,
    ui32 nIterations,
    std::map<ui32, ui32>* nodeToClusterIndex
) {
    std::cerr << "num vertexes " << graph.Vertexes().size() << std::endl;
    TEdgeFunction similarity = graph.CreateEmptyEdgeFunction(-1.f);

    for (ui32 nodeIdx: graph.Vertexes()) {
        TEdge selfEdge{nodeIdx, nodeIdx};
        similarity[selfEdge] = -10000.;
    }
    std::cerr << "finish initializing self edges" << std::endl;

    TEdgeFunction availibility = graph.CreateEmptyEdgeFunction(0.f);
    TEdgeFunction responsibility = graph.CreateEmptyEdgeFunction(0.f);

    for (ui32 iter = 0; iter < nIterations; iter++) {
        std::cerr << "--- iteration " << iter + 1 << std::endl;
        UpdateResponsibility(graph, similarity, availibility, responsibility);
        UpdateAvailibility(graph, similarity, responsibility, availibility);
    }

    for (ui32 nodeIdx: graph.Vertexes()) {
        const auto& adjacents = graph.GetAdjacents(nodeIdx);

        ui32 maxNode = *adjacents.begin();
        float maxValue = 0;
        {
            TEdge firstEdge{nodeIdx, maxNode};
            maxValue = availibility[firstEdge] + responsibility[firstEdge];
        }
        for (ui32 adjacentNode: adjacents) {
            TEdge edge{nodeIdx, adjacentNode};
            const float value = availibility[edge] + responsibility[edge];
            if (value > maxValue) {
                maxValue = value;
                maxNode = adjacentNode;
            }
        }

        nodeToClusterIndex->insert(std::make_pair(nodeIdx, maxNode));
    }
}


bool SortBySecond(const std::pair<ui32,ui32> &a, const std::pair<ui32,ui32> &b) {
    return (a.second < b.second);
}


class TUserToLocations {
public:
    TUserToLocations() {
        MaxLocationId = 0;
    }

    void AddUserLocation(ui32 userId, ui32 location) {
        Locations[userId].push_back(location);
        if (location > MaxLocationId) {
            MaxLocationId = location;
        }

    }

    void SetMaxLocationId(ui32 maxLocationId) {
        MaxLocationId = maxLocationId;
    }

    void SetUserLocations(ui32 userId, const std::vector<ui32>& locations) {
        Locations[userId] = locations;
    }

    void RandomSplitUsers(TUserToLocations* train, TUserToLocations* test) const {
        for (auto& [userId, locations]: Locations) {
            if (static_cast<float>(rand()) / RAND_MAX > 0.33) {
                train->SetUserLocations(userId, locations);
            } else {
                test->SetUserLocations(userId, locations);
            }
        }
        train->SetMaxLocationId(MaxLocationId);
        test->SetMaxLocationId(MaxLocationId);
    }

    std::unordered_map<ui32, std::vector<ui32>> GetTopClusterLocations(const std::map<ui32, ui32>& userToClusterIdx) const {
//        typedef ui32 TLocationId;
        typedef ui32 TClusterId;
        std::unordered_map<TClusterId, std::unordered_map<TLocationId, ui32>> locationToFrequency;

//        std::cout << "Calculating locationsFrequencies" << std::endl;
        for (auto& [userId, locations]: Locations) {
            const ui32 clusterIdx = userToClusterIdx.at(userId);
            for (ui32 location: locations) {
                locationToFrequency[clusterIdx][location] += 1;
            }
        }

        std::unordered_map<ui32, std::vector<ui32>> result;
        for (auto& [clusterIdx, locationsFrequencies]: locationToFrequency) {
            std::vector<std::pair<TLocationId, ui32>> sortedLocations;
            sortedLocations.resize(locationsFrequencies.size());
            std::copy(locationsFrequencies.begin(), locationsFrequencies.end(), sortedLocations.begin());

            std::sort(sortedLocations.begin(), sortedLocations.end(), SortBySecond);
            for (ui32 i = sortedLocations.size() - 11; i < sortedLocations.size() - 1; i++) {
                result[clusterIdx].push_back(sortedLocations[i].first);
            }

//            std::sort(sortedLocations.begin(), sortedLocations.end(), SortBySecond);
//            for (ui32 i = 0; i < std::min(10u, static_cast<ui32>(sortedLocations.size())); i++) {
//                result[clusterIdx].push_back(sortedLocations[i].first);
//            }
        }

//        std::cout << "Result: " << result << std::endl;

        return result;
    }

    std::vector<TLocationId> GetOveralTopClusterLocations(const std::map<ui32, ui32>& userToClusterIdx) const {
//        typedef ui32 TLocationId;
        typedef ui32 TClusterId;
        std::unordered_map<ui32, ui32> locationToFrequency;

//        std::cout << "Calculating locationsFrequencies" << std::endl;
        for (auto& [userId, locations]: Locations) {
            for (ui32 location: locations) {
                locationToFrequency[location] += 1;
            }
        }

        std::vector<TLocationId> ovResult;
        std::vector<std::pair<TLocationId, ui32>> sortedLocations;
        sortedLocations.resize(locationToFrequency.size());
        std::copy(locationToFrequency.begin(), locationToFrequency.end(), sortedLocations.begin());

        std::sort(sortedLocations.begin(), sortedLocations.end(), SortBySecond);
        for (ui32 i = sortedLocations.size() - 11; i < sortedLocations.size() - 1; i++) {
            ovResult.push_back(sortedLocations[i].first);
        }

//        std::sort(sortedLocations.begin(), sortedLocations.end(), SortBySecond);
//        for (ui32 i = 0; i < std::min(100u, static_cast<ui32>(sortedLocations.size())); i++) {
//            ovResult.push_back(sortedLocations[i].first);
//        }

//        for (int i = sortedLocations.size() - 11; i < sortedLocations.size() - 1; i++)
//            std::cout << "Location " << sortedLocations[i].first << " - " << locationToFrequency[sortedLocations[i].first] << std::endl;

//        std::cout << "Overal result: " << ovResult << std::endl;

        return ovResult;
    }

    const std::unordered_map<ui32, std::vector<ui32>>& GetLocations() const {
        return Locations;
    }

    ui32 GetNumUsers() const {
        return Locations.size();
    }

private:
    std::unordered_map<ui32, std::vector<ui32>> Locations;
    ui32 MaxLocationId;
};


float Accuracy(
    const std::unordered_map<ui32, std::vector<ui32>>& topLocations,
    const std::map<ui32, ui32>& userToClusterIdx,
    const TUserToLocations& test
) {
    ui32 num_misses = 0;
    float overallSum = 0.;

    for (auto& [user, locations]: test.GetLocations()) {
        const ui32 clusterIdx = userToClusterIdx.at(user);
        if (topLocations.find(clusterIdx) == topLocations.end()) {
            num_misses++;
            continue;
        }
        const auto& topClusterLocations = topLocations.at(clusterIdx);
        float curSum = 0.;

        for (ui32 topLocation: topClusterLocations) {
            for (ui32 userLocation: locations) {
                if (userLocation == topLocation) {
                    curSum += 1.;
                    break;
                }
            }
        }

        curSum /= 10.;
        overallSum += curSum;
    }

//    std::cout << "Num misses " << num_misses << std::endl;

    overallSum /= static_cast<float>(test.GetNumUsers() - num_misses);
    return overallSum;
}

float AccuracyOveral(
        const std::vector<TLocationId>& topLocations,
        const std::map<ui32, ui32>& userToClusterIdx,
        const TUserToLocations& test
) {
    ui32 num_misses = 0;
    float overallSum = 0.;

    for (auto& [user, locations]: test.GetLocations()) {
        const ui32 clusterIdx = userToClusterIdx.at(user);

        float curSum = 0.;

        for (ui32 location: topLocations) {
            for (ui32 userLocation: locations) {
                if (userLocation == location) {
                    curSum += 1.;
                    break;
                }
            }
        }

        curSum /= 10.;
        overallSum += curSum;
    }

    //std::cout << "Num misses " << num_misses << std::endl;

    overallSum /= static_cast<float>(test.GetNumUsers() - num_misses);
    return overallSum;
}


int main(int argc, char* argv[]) {
    using namespace std;

    int numIterations = atoi(argv[1]);
    char* graphFileName = argv[2];
    char* locationFileName = argv[3];

    ifstream graphFile(graphFileName);

    TGraph graph;

    string line;
    while (getline(graphFile, line)) {
        stringstream ss(line);
        ui32 node1;
        ui32 node2;
        ss >> node1 >> node2;

        graph.AddEdge(node1, node2);
    }

    std::cerr << "Finish read graph" << std::endl;

    std::map<ui32, ui32> nodeToClusterIndex;
    AffinityPropagation(graph, numIterations, &nodeToClusterIndex);

    std::map<ui32, ui32> clusterSizes;
    for (auto& [node, cluster]: nodeToClusterIndex) {
        clusterSizes[cluster] += 1;
    }

    std::cout << "----- Results analysis -----" << std::endl;
    std::cout << "- Number of clusters: " << clusterSizes.size() << endl;

    {
        std::vector<ui32> clusterIds;
        clusterIds.resize(clusterSizes.size());
        {
            ui32 i = 0;
            for (auto& [clusterId, clusterSize]: clusterSizes) {
                clusterIds[i] = clusterId;
                i++;
            }
        }

        std::sort(
            clusterIds.begin(),
            clusterIds.end(),
            [&clusterSizes](ui32 leftId, ui32 rightId) {
                return clusterSizes[leftId] < clusterSizes[rightId];
            }
        );

        std::cout << "- Min cluster size: " << clusterSizes[clusterIds[0]] << endl;
        std::cout << "- 20 percentile cluster size: " << clusterSizes[clusterIds[static_cast<ui32>(clusterIds.size() * 0.2)]] << endl;
        std::cout << "- Median cluster size: " << clusterSizes[clusterIds[clusterIds.size() / 2]] << endl;
        std::cout << "- 80 percentile cluster size: " << clusterSizes[clusterIds[static_cast<ui32>(clusterIds.size() * 0.8)]] << endl;
        std::cout << "- Max cluster size: " << clusterSizes[clusterIds[clusterIds.size() - 1]] << endl;

        std::cout << "----- Max cluster sizes -----" << std::endl;
        for (int i = 0; i < 20; i++)
            std::cout << i + 1 << " cluster size: " << clusterSizes[clusterIds[clusterIds.size() - 1 - i]] << endl;
    }

    TUserToLocations UserToLocations;
    ifstream locationsFile(locationFileName);
    while (getline(locationsFile, line)) {
        stringstream ss(line);

        ui32 user;
        std::string checkIn;
        float latitude;
        float longitude;
        ui32 locationId;

        ss >> user >> checkIn >> latitude >> longitude >> locationId;
        UserToLocations.AddUserLocation(user, locationId);
    }

    TUserToLocations Train;
    TUserToLocations Test;
    std::cout << "SplitUsers" << std::endl;
    UserToLocations.RandomSplitUsers(&Train, &Test);

    float accuracy = Accuracy(Train.GetTopClusterLocations(nodeToClusterIndex), nodeToClusterIndex, Test);
    float accuracy1 = AccuracyOveral(Train.GetOveralTopClusterLocations(nodeToClusterIndex), nodeToClusterIndex, Test);

    std::cout << "----- Accuracy analysis -----" << std::endl;
    std::cerr << "Accuracy for top–10 locations in cluster: " << accuracy << std::endl;
    std::cerr << "Accuracy for overal top-10 locations: " << accuracy1 << std::endl;

    return 0;
}
