// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>

#include "../utils/NSGDist.h"
#include "../utils/beamSearch.h"
#include "../utils/check_nn_recall.h"
#include "../utils/parse_results.h"
#include "../utils/mips_point.h"
#include "../utils/euclidian_point.h"
#include "../utils/stats.h"
#include "../utils/types.h"
#include "../utils/graph.h"
#include "../utils/graph_disk.h"
#include "index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

#include <chrono>
#include <sys/resource.h>
#include <iostream>

// 메모리 사용량을 출력하는 함수
long get_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // 현재 메모리 사용량 (KB)
}

template<typename Point, typename PointRange, typename QPointRange, typename indexType>
void ANN_(Graph<indexType> &G, long k, BuildParams &BP,
          PointRange &Query_Points, QPointRange &Q_Query_Points,
          groundTruth<indexType> GT, char *res_file,
          bool graph_built, PointRange &Points, QPointRange &Q_Points) {
}

template<typename Point, typename PointRange_, typename indexType>
void ANN(Graph<indexType> &G, long k, BuildParams &BP,
         PointRange_ &Query_Points,
         groundTruth<indexType> GT, char *res_file,
         bool graph_built, PointRange_ &Points) {
}

template<typename Point, typename PointRange_, typename indexType, typename pointType>
double ANN_disk_partition(char *input_file, char *graph_file, char *out_file, bool graph_built,
                        BuildParams &BP, stats<indexType> &BuildStats, int num_points, int d, int partition, double io_time) {
  int start;
  int end;

  for (int i = 0; i < partition; i++) {
    // parlay::internal::timer t("ANN");

    start = i * num_points / partition;
    end = (i + 1) * num_points / partition;
    std::cout << "Reading partition " << i << " from " << start << " to " << end << std::endl;
    auto start_io_time = std::chrono::high_resolution_clock::now(); // 타이머 시작
    PointRange<pointType, Point> Points = PointRange<pointType, Point>(input_file, start, end);
    auto end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
    std::chrono::duration<double> io_duration = end_io_time - start_io_time;
    io_time += io_duration.count(); 
    // std::cout << "Read " << Points.size() << " points" << std::endl;

    Graph_disk<unsigned int> G;
    G = Graph_disk<unsigned int>(BP.max_degree(), Points.size());
    // if(graph_file == NULL) G = Graph_disk<unsigned int>(BP.max_degree(), Points.size());
    // else G = Graph_disk<unsigned int>(graph_file);

    bool verbose = BP.verbose;
    using findex = knn_index<PointRange_, indexType>;
    findex I(BP);
    indexType start_point;
    double idx_time;
    if(graph_built){
      idx_time = 0;
      start_point = 0;
    } else{
      I.build_index_disk(G, Points, BuildStats, BP.alpha, start, true, 2, .02, true, true);
      // I.build_index(G, Points, BuildStats);
      start_point = I.get_start();
      // start_point = start;
      // idx_time = t.next_time();
    }
    // std::cout << "start index = " << start_point << std::endl;

    // print graph examples
    // for (int i = 0; i < 10; i++) {
    //   std::cout << "Example of vertex " << i << "(total " << G[i].size() << ")" << std::endl;
    //   for (int j = 0; j < G[i].size(); j++) {
    //     std::cout << "(" << G[i][j].first << " " << G[i][j].second << ") ";
    //   }
    //   std::cout << std::endl;
    // }

    std::string name = "Vamana";
    std::string params =
      "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
    // auto [avg_deg, max_deg] = graph_stats_(G);
    auto od = parlay::delayed_seq<size_t>(
        G.size(), [&](size_t i) { return G[i].size(); });
    size_t j = parlay::max_element(od) - od.begin();
    int max_deg = od[j];
    size_t sum1 = parlay::reduce(od);
    double avg_deg = sum1 / ((double)G.size());
    auto vv = BuildStats.visited_stats();
    // std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
    //           << std::endl;

    Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
    G_.print();


    start_io_time = std::chrono::high_resolution_clock::now(); // 타이머 시작
    if(out_file != NULL) {
      std::string outfile(out_file);
      outfile += "_sub_" + std::to_string(i);
      std::cout << "Saving graph to " << outfile << std::endl;
      // G.save(outfile.c_str());
      G.save_subgraph(const_cast<char*>(outfile.c_str()));
    }
    end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
    io_duration = end_io_time - start_io_time;
    io_time += io_duration.count(); 

  }
  return io_time;
}


template<typename Point, typename PointRange_, typename indexType, typename pointType>
double ANN_disk_merge(char *input_file, char *graph_file, bool graph_built, BuildParams &BP, stats<indexType> &BuildStats, 
                    int num_points, int d, int partition, double io_time) {
  int partition_size = num_points / partition;
  int p1;
  int p2;
  PointRange<pointType, Point> Points1;
  PointRange<pointType, Point> Points2;
  Graph_disk<unsigned int> G1;
  Graph_disk<unsigned int> G2;

  // 병합 여부를 기록할 Merged 행렬 초기화
  std::vector<std::vector<bool>> Merged(partition, std::vector<bool>(partition, false));
  for (int i = 0; i < partition; i++) {
    Merged[i][i] = true;
  }

  // Read First Partition and Graph
  p1 = 0;
  int interval = 1;
  std::string graphfile(graph_file);
  auto start_io_time = std::chrono::high_resolution_clock::now(); // 타이머 시작
  std::cout << "(Partition 1) Reading partition " << p1 
            << " from " << p1 * partition_size << " to " << (p1 + 1) * partition_size << std::endl;
  Points1 = PointRange<pointType, Point>(input_file, p1 * partition_size, (p1 + 1) * partition_size);
  G1 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
  auto end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
  std::chrono::duration<double> io_duration = end_io_time - start_io_time;
  io_time += io_duration.count(); 

  while (true) {
    p2 = (p1 + interval) % partition;
    if (Merged[p1][p2]) {
      interval++;
      continue;
    }

    // Read Second Partition and Graph
    start_io_time = std::chrono::high_resolution_clock::now(); // 타이머 시작
    std::cout << "(Partition 2) Reading partition " << p2 
              << " from " << p2 * partition_size << " to " << (p2 + 1) * partition_size << std::endl;
    Points2 = PointRange<pointType, Point>(input_file, p2 * partition_size, (p2 + 1) * partition_size);
    G2 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p2)).c_str()));
    end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
    io_duration = end_io_time - start_io_time;
    io_time += io_duration.count();

    bool verbose = BP.verbose;
    using findex = knn_index<PointRange_, indexType>;
    findex I(BP);
    indexType start_point;
    // double idx_time;
    if(graph_built){
      // idx_time = 0;
      start_point = 0;
    } else{
      I.merge_index_disk(G1, G2, Points1, Points2, p1, p2, partition_size,
                        BuildStats, BP.alpha, true, true, true);
      // std::cout << "Merged " << " from " << p1 << " to " << p2 << std::endl;
      start_point = I.get_start();
      // idx_time = t.next_time();
    }

    // std::cout << "start index = " << start_point << std::endl;


    // std::string name = "Vamana";
    // std::string params =
    //   "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
    // // auto [avg_deg, max_deg] = graph_stats_(G);
    // auto od = parlay::delayed_seq<size_t>(
    //     G1.size(), [&](size_t i) { return G[i].size(); });
    // size_t j = parlay::max_element(od) - od.begin();
    // int max_deg = od[j];
    // size_t sum1 = parlay::reduce(od);
    // double avg_deg = sum1 / ((double)G1.size());
    // auto vv = BuildStats.visited_stats();
    // std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
    //           << std::endl;

    // Graph_ G_(name, params, G1.size(), avg_deg, max_deg, idx_time);
    // G_.print();

    // Write graph 1
    // std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p1) + "_updated" << std::endl << std::endl;
    // G1.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1) + "_updated").c_str()));
    start_io_time = std::chrono::high_resolution_clock::now(); // 타이머 시작
    std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p1) << std::endl << std::endl;
    G1.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
    end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
    io_duration = end_io_time - start_io_time;
    io_time += io_duration.count();

    Merged[p1][p2] = true;
    Merged[p2][p1] = true;
    p1 = p2;
    // p1에 대한 병합된 파티션 개수가 전체 파티션 수와 동일한지 확인
    int merge_count = std::count(Merged[p1].begin(), Merged[p1].end(), true);
    if (merge_count == partition) {
      std::cout << "All merges for partition " << p1 << " are completed. Exiting..." << std::endl;
      break;
    }

    std::swap(Points1, Points2);
    std::swap(G1, G2);

  }

  // print merged
  for (int i = 0; i < partition; i++) {
    for (int j = 0; j < partition; j++) {
      std::cout << Merged[i][j] << " ";
    }
    std::cout << std::endl;
  }

  // Write graph 2
  // std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p2) + "_updated" << std::endl;
  // G2.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p2) + "_updated").c_str()));
  start_io_time = std::chrono::high_resolution_clock::now(); // 타이머 시작
  std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p2) << std::endl;
  G2.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p2)).c_str()));
  end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
  io_duration = end_io_time - start_io_time;
  io_time += io_duration.count();

  return io_time;

}

// template<typename Point, typename PointRange_, typename indexType, typename pointType>
// double ANN_disk_partition_batch(char *input_file, char *graph_file, bool graph_built, BuildParams &BP, stats<indexType> &BuildStats, 
//                     int num_points, int d, int partition, double io_time, double max_fraction, double merge_ratio) {
//   size_t partition_size = num_points / partition;
//   size_t p1;
//   size_t p2;
//   PointRange<pointType, Euclidian_Point<pointType>> Points;
  

//   for (int i=0; i < partition; i++) {
//     p1 = i;
//     p2 = (i + 1) % partition;
//     if (i == 0) {
//       Points = PointRange<pointType, Euclidian_Point<pointType>>(input_file, p1 * partition_size, (p2 + 1) * partition_size);
//     } else {
//       Points = PointRange<pointType, Euclidian_Point<pointType>>(input_file, p1 * partition_size, (p1 + 1) * partition_size);
//       // Points.append(input_file, p2 * partition_size, (p2 + 1) * partition_size);
//     }
//     std::cout << "Reading partition " << p1 << " from " << p1 * partition_size << " to " << (p2 + 1) * partition_size << std::endl;
//     std::cout << "Total points: " << Points.size() << std::endl;
//     std::cout << "Points exampls" << std::endl;
//     for (int j = 0; j < 3; j++) {
//       std::cout << "Point " << p1 * partition_size + j << ": ";
//       for (int d = 0; d < Points.get_dims(); d++) {
//         std::cout << static_cast<int>(Points[j][d]) << " ";
//       }
//       std::cout << std::endl;
//     }

//     std::cout << std::endl;
//   }

//   return io_time;

//   // Graph_disk<unsigned int> G1;
//   // Graph_disk<unsigned int> G2;

//   // // 병합 여부를 기록할 Merged 행렬 초기화
//   // std::vector<std::vector<bool>> Merged(partition, std::vector<bool>(partition, false));
//   // // for (int i = 0; i < partition; i++) {
//   // //   Merged[i][i] = true;
//   // // }

//   // bool verbose = BP.verbose;
//   // using findex = knn_index<PointRange_, indexType>;
//   // findex I(BP);
//   // indexType start_point;

//   // std::string graphfile(graph_file);

//   // for (int i; i < partition; i++) {
//   //   std::cout << "(Partition 1) Reading partition " << i
//   //             << " from " << i * partition_size << " to " << (i + 1) * partition_size << std::endl;
//   //   auto start_io_time = std::chrono::high_resolution_clock::now();
//   //   Points1 = PointRange<pointType, Euclidian_Point<pointType>>(input_file, i * partition_size, (i + 1) * partition_size);
//   //   // G1 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
//   //   // G1 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1) + "_first").c_str()));
//   //   G1 = Graph_disk<unsigned int>(BP.max_degree(), Points1.size());
//   //   auto end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
//   //   std::chrono::duration<double> io_duration = end_io_time - start_io_time;
//   //   io_time += io_duration.count();  // I/O 시간 누적
//   //   Merged[i][i] = true;

//   //   I.build_index_disk(G1, Points1, BuildStats, BP.alpha, i * partition_size, true, 2, .02, true, true);
    
//   //   // Write graph 1
//   //   start_io_time = std::chrono::high_resolution_clock::now();
//   //   std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p1) << std::endl << std::endl;
//   //   G1.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
//   //   end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
//   //   io_duration = end_io_time - start_io_time;
//   //   io_time += io_duration.count();  // I/O 시간 누적
//   // }

//   // // Read First partition and Build Graph
//   // for (int i; i < partition; i++) {
//   //   p1 = i;
    
//   //   std::cout << "(Partition 1) Reading partition " << p1 
//   //             << " from " << p1 * partition_size << " to " << (p1 + 1) * partition_size << std::endl;
//   //   auto start_io_time = std::chrono::high_resolution_clock::now();
//   //   Points1 = PointRange<pointType, Euclidian_Point<pointType>>(input_file, p1 * partition_size, (p1 + 1) * partition_size);
//   //   G1 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
//   //   // G1 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1) + "_first").c_str()));
//   //   G1 = Graph_disk<unsigned int>(BP.max_degree(), Points1.size());
//   //   auto end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
//   //   std::chrono::duration<double> io_duration = end_io_time - start_io_time;
//   //   io_time += io_duration.count();  // I/O 시간 누적
//   //   // Merged[p1][p1] = true;

//   //   // I.build_index_disk(G1, Points1, BuildStats, BP.alpha, p1 * partition_size, true, 2, .02, true, true);

//   //   p2 = (p1 + 1) % partition;
//   //   while (p2 != p1) {
//   //     // Read Second Partition and Graph
//   //     std::cout << std::endl;
//   //     std::cout << "(Partition 2) Reading partition " << p2 
//   //               << " from " << p2 * partition_size << " to " << (p2 + 1) * partition_size << std::endl;
//   //     start_io_time = std::chrono::high_resolution_clock::now();
//   //     Points2 = PointRange<pointType, Euclidian_Point<pointType>>(input_file, p2 * partition_size, (p2 + 1) * partition_size);
//   //     if (Merged[p2][p2]) {
//   //       G2 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p2)).c_str()));
//   //     } else {
//   //       G2 = Graph_disk<unsigned int>(BP.max_degree(), Points2.size());
//   //     }
//   //     end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
//   //     io_duration = end_io_time - start_io_time;
//   //     io_time += io_duration.count();  // I/O 시간 누적

//   //     findex I(BP);
//   //     double base = 2;
//   //     double alpha = BP.alpha;
      
//   //     I.build_and_merge_index(G1, G2, Points1, Points2, p1, p2, partition_size, BuildStats, alpha, base, max_fraction,
//   //                             false, true, true);

//   //     // print graph examples
//   //     std::cout << "Graph " << p1 << std::endl;
//   //     for (int i = 0; i < 3; i++) {
//   //       std::cout << "Example of vertex " << i << "(total " << G1[i].size() << ")" << std::endl;
//   //       for (int j = 0; j < G1[i].size(); j++) {
//   //         std::cout << "(" << G1[i][j].first << " " << G1[i][j].second << ") ";
//   //       }
//   //       std::cout << std::endl;
//   //     }
//   //     std::cout << "..." << std::endl;
//   //     for (int i = G1.size()-3; i < G1.size(); i++) {
//   //       std::cout << "Example of vertex " << i << "(total " << G1[i].size() << ")" << std::endl;
//   //       for (int j = 0; j < G1[i].size(); j++) {
//   //         std::cout << "(" << G1[i][j].first << " " << G1[i][j].second << ") ";
//   //       }
//   //       std::cout << std::endl;
//   //     }

//   //     std::cout << "Graph " << p2 << std::endl;
//   //     for (int i = 0; i < 3; i++) {
//   //       std::cout << "Example of vertex " << i << "(total " << G2[i].size() << ")" << std::endl;
//   //       for (int j = 0; j < G2[i].size(); j++) {
//   //         std::cout << "(" << G2[i][j].first << " " << G2[i][j].second << ") ";
//   //       }
//   //       std::cout << std::endl;
//   //     }
//   //     std::cout << "..." << std::endl;
//   //     for (int i = G2.size()-3; i < G2.size(); i++) {
//   //       std::cout << "Example of vertex " << i << "(total " << G2[i].size() << ")" << std::endl;
//   //       for (int j = 0; j < G2[i].size(); j++) {
//   //         std::cout << "(" << G2[i][j].first << " " << G2[i][j].second << ") ";
//   //       }
//   //       std::cout << std::endl;
//   //     }

//   //     auto vv = BuildStats.visited_stats();
//   //     std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1] << ", Max visited: " << vv[2]
//   //               << std::endl;

//   //     // Write graph 2
//   //     start_io_time = std::chrono::high_resolution_clock::now();
//   //     std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p2) << std::endl;
//   //     G2.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p2)).c_str()));
//   //     end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
//   //     io_duration = end_io_time - start_io_time;
//   //     io_time += io_duration.count();  // I/O 시간 누적

//   //     p2 = (p2 + 1) % partition;
//   //   }

//   //   // Write graph 1
//   //   start_io_time = std::chrono::high_resolution_clock::now();
//   //   std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p1) << std::endl << std::endl;
//   //   G1.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
//   //   end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
//   //   io_duration = end_io_time - start_io_time;
//   //   io_time += io_duration.count();  // I/O 시간 누적

//   // }

//   // return io_time;

// }



template<typename Point, typename PointRange_, typename indexType, typename pointType>
double ANN_disk_partition_batch(char *input_file, char *graph_file, bool graph_built, BuildParams &BP, stats<indexType> &BuildStats, 
                    int num_points, int d, int partition, double io_time, double max_fraction, double merge_ratio) {
  int partition_size = num_points / partition;
  int p1;
  int p2;
  PointRange<pointType, Point> Points1;
  PointRange<pointType, Point> Points2;
  Graph_disk<unsigned int> G1;
  Graph_disk<unsigned int> G2;

  // 병합 여부를 기록할 Merged 행렬 초기화
  std::vector<std::vector<bool>> Merged(partition, std::vector<bool>(partition, false));

  bool verbose = BP.verbose;
  using findex = knn_index<PointRange_, indexType>;
  findex I(BP);
  // indexType start_point;

  // Read First partition and Build Graph
  p1 = 0;
  int interval = 1;
  std::string graphfile(graph_file);
  std::cout << "(Partition 1) Reading partition " << p1 
            << " from " << p1 * partition_size << " to " << (p1 + 1) * partition_size << std::endl;
  auto start_io_time = std::chrono::high_resolution_clock::now();
  Points1 = PointRange<pointType, Point>(input_file, p1 * partition_size, (p1 + 1) * partition_size);
  // for (int i=0; i < Points1.size(); i++) Points1[i].normalize();
  // std::cout << "Points1 exampls(1048576)" << std::endl;
  // for (int d = 0; d < Points1.get_dims(); d++) {
  //   std::cout << static_cast<pointType>(Points1[1048576-(p1 * partition_size)][d]) << " ";
  // }
  // std::cout << std::endl;
  // for (int j = 0; j < 3; j++) {
  //   std::cout << "Point " << p1 * partition_size + j << ": ";
  //   for (int d = 0; d < Points1.get_dims(); d++) {
  //     std::cout << static_cast<int>(Points1[j][d]) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // G1 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
  // G1 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1) + "_first").c_str()));
  G1 = Graph_disk<unsigned int>(BP.max_degree(), Points1.size());
  auto end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
  std::chrono::duration<double> io_duration = end_io_time - start_io_time;
  io_time += io_duration.count();  // I/O 시간 누적

  double alpha = BP.alpha;
  std::cout << "Building index for partition " << p1 << std::endl;  
  I.build_index_disk(G1, Points1, BuildStats, alpha, p1 * partition_size, true, 2, .02, true, true);

  // print graph examples
  // std::cout << "Graph " << p1 << std::endl;
  // for (int i = 0; i < 3; i++) {
  //   std::cout << "Example of vertex " << i << "(total " << G1[i].size() << ")" << std::endl;
  //   for (int j = 0; j < G1[i].size(); j++) {
  //     std::cout << "(" << G1[i][j].first << " " << G1[i][j].second << ") ";
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p1) + "_first" << std::endl << std::endl;
  // G1.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1) + "_first").c_str()));


  if (partition == 1) {
    // Write graph 1
    start_io_time = std::chrono::high_resolution_clock::now();
    std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p1) << std::endl << std::endl;
    G1.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
    end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
    io_duration = end_io_time - start_io_time;
    io_time += io_duration.count();  // I/O 시간 누적
    
    std::cout << "Only one partition. Exiting..." << std::endl;
    return io_time;
  }

  Merged[p1][p1] = true;

  while (true) {
    p2 = (p1 + interval) % partition;
    if (p1 == p2) break;
    if (Merged[p2][p1]) {
      // interval++;
      interval = interval * 2;
      alpha = 1.0;
      // alpha = std::max(alpha * 1.15, BP.alpha);
      continue;
    }
    
    // Read Second Partition and Graph
    std::cout << std::endl;
    std::cout << "(Partition 2) Reading partition " << p2 
              << " from " << p2 * partition_size << " to " << (p2 + 1) * partition_size << std::endl;
    start_io_time = std::chrono::high_resolution_clock::now();
    Points2 = PointRange<pointType, Point>(input_file, p2 * partition_size, (p2 + 1) * partition_size);
    // for (int i=0; i < Points2.size(); i++) Points2[i].normalize();
    // std::cout << "Points2 exampls (184568420)" << std::endl;
    // for (int d = 0; d < Points2.get_dims(); d++) {
    //   std::cout << static_cast<pointType>(Points2[184568420-(p2 * partition_size)][d]) << " ";
    // }
    // std::cout << std::endl;

    // for (int j = 0; j < 3; j++) {
    //   std::cout << "Point " << p2 * partition_size + j << ": ";
    //   for (int d = 0; d < Points2.get_dims(); d++) {
    //     std::cout << static_cast<pointType>(Points2[j][d]) << " ";
    //   }
    //   std::cout << std::endl;
    // }
    if (Merged[p2][p2]) {
      G2 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p2)).c_str()));
    } else {
      G2 = Graph_disk<unsigned int>(BP.max_degree(), Points2.size());
    }
    end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
    io_duration = end_io_time - start_io_time;
    io_time += io_duration.count();  // I/O 시간 누적

    findex I(BP);
    double base = 2;
    // double max_fraction = .5;
    // double alpha = BP.alpha;
    
    I.build_and_merge_index(G1, G2, Points1, Points2, p1, p2, partition_size, BuildStats, alpha, base, max_fraction,
                            true, true, true);
    // idx_time = t.next_time();

    // // print graph examples
    // std::cout << "Graph " << p1 << std::endl;
    // for (int i = 0; i < 3; i++) {
    //   std::cout << "Example of vertex " << i << "(total " << G1[i].size() << ")" << std::endl;
    //   for (int j = 0; j < G1[i].size(); j++) {
    //     std::cout << "(" << G1[i][j].first << " " << G1[i][j].second << ") ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "..." << std::endl;
    // for (int i = G1.size()-3; i < G1.size(); i++) {
    //   std::cout << "Example of vertex " << i << "(total " << G1[i].size() << ")" << std::endl;
    //   for (int j = 0; j < G1[i].size(); j++) {
    //     std::cout << "(" << G1[i][j].first << " " << G1[i][j].second << ") ";
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "Graph " << p2 << std::endl;
    // for (int i = 0; i < 3; i++) {
    //   std::cout << "Example of vertex " << i << "(total " << G2[i].size() << ")" << std::endl;
    //   for (int j = 0; j < G2[i].size(); j++) {
    //     std::cout << "(" << G2[i][j].first << " " << G2[i][j].second << ") ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "..." << std::endl;
    // for (int i = G2.size()-3; i < G2.size(); i++) {
    //   std::cout << "Example of vertex " << i << "(total " << G2[i].size() << ")" << std::endl;
    //   for (int j = 0; j < G2[i].size(); j++) {
    //     std::cout << "(" << G2[i][j].first << " " << G2[i][j].second << ") ";
    //   }
    //   std::cout << std::endl;
    // }

    // std::string name = "Vamana";
    // std::string params =
    //   "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
    // // auto [avg_deg, max_deg] = graph_stats_(G);
    // auto od = parlay::delayed_seq<size_t>(
    //     G1.size(), [&](size_t i) { return G[i].size(); });
    // size_t j = parlay::max_element(od) - od.begin();
    // int max_deg = od[j];
    // size_t sum1 = parlay::reduce(od);
    // double avg_deg = sum1 / ((double)G1.size());
    // auto vv = BuildStats.visited_stats();
    // std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
    //           << std::endl;

    // Graph_ G_(name, params, G1.size(), avg_deg, max_deg, idx_time);
    // G_.print();

    auto vv = BuildStats.visited_stats();
    std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1] << ", Max visited: " << vv[2]
              << std::endl;

    Merged[p1][p2] = true;
    Merged[p2][p1] = true;
    Merged[p2][p2] = true;

    // Write graph 1
    start_io_time = std::chrono::high_resolution_clock::now();
    std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p1) << std::endl << std::endl;
    G1.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
    end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
    io_duration = end_io_time - start_io_time;
    io_time += io_duration.count();  // I/O 시간 누적

    p1 = p2;

    // if (p1 == 0) {
    //   interval *= 2;
    //   alpha = std::max(alpha * 0.9, 1.0);
    // }

    // 전체 Merged 배열에서 true의 비율 계산
    int total_entries = partition * partition;
    int true_count = 0;
    for (int i = 0; i < partition; i++) {
      true_count += std::count(Merged[i].begin(), Merged[i].end(), true);
    }
    
    // true의 비율이 특정비율에 도달하면 종료
    double true_ratio = static_cast<double>(true_count) / total_entries;
    if (true_ratio >= merge_ratio) {
      std::cout << "True ratio " << true_ratio << " has reached "<< merge_ratio * 100 << "% of total. Exiting..." << std::endl;
      break;
    }

    int merge_count = std::count(Merged[p1].begin(), Merged[p1].end(), true);
    if (merge_count == partition) {
      std::cout << "All merges for partition " << p1 << " are completed. Exiting..." << std::endl;
      // Write graph 2
      start_io_time = std::chrono::high_resolution_clock::now();
      std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p1) << std::endl;
      G1.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
      end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
      io_duration = end_io_time - start_io_time;
      io_time += io_duration.count();  // I/O 시간 누적

      p1 = (p1 + 1) % partition;
      std::cout << "(Partition 2) Reading partition " << p1 
              << " from " << p1 * partition_size << " to " << (p1 + 1) * partition_size << std::endl;
      start_io_time = std::chrono::high_resolution_clock::now();
      Points1 = PointRange<pointType, Point>(input_file, p1 * partition_size, (p1 + 1) * partition_size);
      // std::cout << "Points1 exampls" << std::endl;
      // for (int j = 0; j < 3; j++) {
      //   std::cout << "Point " << p1 * partition_size + j << ": ";
      //   for (int d = 0; d < Points1.get_dims(); d++) {
      //     std::cout << static_cast<int>(Points1[j][d]) << " ";
      //   }
      //   std::cout << std::endl;
      // }
      if (Merged[p1][p1]) {
        G1 = Graph_disk<unsigned int>(const_cast<char*>((graphfile + "_sub_" + std::to_string(p1)).c_str()));
      } else {
        G1 = Graph_disk<unsigned int>(BP.max_degree(), Points1.size());
      }
      end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
      io_duration = end_io_time - start_io_time;
      io_time += io_duration.count();  // I/O 시간 누적
      continue;
    }

    std::swap(Points1, Points2);
    std::swap(G1, G2);

  }

  // print merged
  for (int i = 0; i < partition; i++) {
    for (int j = 0; j < partition; j++) {
      std::cout << Merged[i][j] << " ";
    }
    std::cout << std::endl;
  }

  // Write graph 2
  start_io_time = std::chrono::high_resolution_clock::now();
  std::cout << "Saving graph to " << graphfile + "_sub_" + std::to_string(p2) << std::endl;
  G2.save_subgraph(const_cast<char*>((graphfile + "_sub_" + std::to_string(p2)).c_str()));
  end_io_time = std::chrono::high_resolution_clock::now(); // 타이머 종료
  io_duration = end_io_time - start_io_time;
  io_time += io_duration.count();  // I/O 시간 누적

  return io_time;

}


template<typename Point, typename PointRange_, typename indexType, typename pointType>
void ANN_disk(char *input_file, char *graph_file, char *out_file, char *res_file, char *query_file, char *gt_file, bool graph_built,
              long k, BuildParams &BP, int partition, bool partition_batch, bool disk_search, double max_fraction, double merge_ratio) {
  long max_memory_usage = get_memory_usage();
  parlay::internal::timer t("ANN");
  double idx_time;

  std::ifstream reader(input_file);
  //read num points and max degree
  unsigned int num_points;
  unsigned int d;
  reader.read((char*)(&num_points), sizeof(unsigned int));
  reader.read((char*)(&d), sizeof(unsigned int));

  stats<unsigned int> BuildStats(num_points);

  double io_time = 0.0;

  // if (partition_batch) {
  //   io_time = ANN_disk_partition_batch<Point, PointRange_, indexType, pointType>(input_file, out_file, graph_built, BP, BuildStats, num_points, d, partition, io_time, max_fraction, merge_ratio);
  //   // current memory usage
  //   long current_memory_usage = get_memory_usage();
  //   if (current_memory_usage > max_memory_usage) {
  //       max_memory_usage = current_memory_usage;
  //   }
  // } else {
  //   // build graph partition
  //   io_time = ANN_disk_partition<Point, PointRange_, indexType, pointType>(input_file, graph_file, out_file, graph_built, BP, BuildStats, num_points, d, partition, io_time);
  //   long current_memory_usage = get_memory_usage();
  //   if (current_memory_usage > max_memory_usage) {
  //       max_memory_usage = current_memory_usage;
  //   }
  //   // merge graph partition
  //   io_time = ANN_disk_merge<Point, PointRange_, indexType, pointType>(input_file, out_file, graph_built, BP, BuildStats, num_points, d, partition, io_time);
  //   current_memory_usage = get_memory_usage();
  //   if (current_memory_usage > max_memory_usage) {
  //       max_memory_usage = current_memory_usage;
  //   }
  // }
  // 전체 I/O 비용 출력
  std::cout << "Total I/O time: " << io_time << " seconds" << std::endl;
  // 최종적으로 기록된 최대 메모리 사용량 출력
  std::cout << "Max memory usage during batch insert: " << max_memory_usage << " KB" << std::endl;
    
  idx_time = t.next_time();

  // std::cout << "vamana graph built with " << num_points 
            // << " points and parameters R = " << std::to_string(BP.R) << ", L = " << std::to_string(BP.L) << std::endl;
    // std::cout << "Graph has average degree " << avg_deg
              // << " and maximum degree " << max_deg << std::endl;
  std::cout << "Graph built in " << idx_time << " seconds" << std::endl;

  Graph<indexType> G; 
  G = Graph<indexType>(out_file, num_points, BP.max_degree(), partition);
  // G = Graph<indexType>(out_file);

  G.save(out_file);
  // G.save_vamana(out_file);
  
  PointRange<pointType, Point> Points = PointRange<pointType, Point>(input_file);

  // // write centroids and medoids
  // std::string centroids_file(out_file);
  // centroids_file += "_diskann_disk.index_centroids.bin";
  // std::cout << "New file name: " << centroids_file << std::endl;
  // std::string medoids_file(out_file);
  // medoids_file += "_diskann_disk.index_medoids.bin";
  // std::cout << "New file name: " << medoids_file << std::endl;

  // // Compute medoids
  // std::vector<uint32_t> medoids;
  // size_t interval = num_points / partition;
  // for (int i = 0; i < partition; i++) {
  //     medoids.push_back(static_cast<uint32_t>(i * interval));
  // }

  // // Save index_medoids.bin
  // std::ofstream medoids_writer(medoids_file, std::ios::binary);
  // if (!medoids_writer.is_open()) {
  //     std::cerr << "Error: Cannot open output file: " << medoids_file << std::endl;
  //     return;
  // }
  // uint32_t num_medoids = static_cast<uint32_t>(medoids.size());
  // uint32_t dimensions_medoids = 1;
  // medoids_writer.write(reinterpret_cast<const char*>(&num_medoids), sizeof(uint32_t));
  // medoids_writer.write(reinterpret_cast<const char*>(&dimensions_medoids), sizeof(uint32_t));
  // medoids_writer.write(reinterpret_cast<const char*>(medoids.data()), medoids.size() * sizeof(uint32_t));
  // medoids_writer.close();
  // std::cout << "index_medoids.bin saved successfully." << std::endl;

  // // Allocate memory for centroids
  // float* centroids = new float[partition * d];
  // if (!centroids) {
  //     std::cerr << "Error: Memory allocation failed for centroids." << std::endl;
  //     return;
  // }

  // // Fill centroids with data
  // for (int i = 0; i < partition; i++) {
  //     const auto& point_data = Points[medoids[i]];
  //     for (unsigned int j = 0; j < d; j++) {
  //         centroids[i * d + j] = point_data[j];
  //     }
  // }

  // // Save index_centroids.bin
  // std::ofstream centroids_writer(centroids_file, std::ios::binary);
  // if (!centroids_writer.is_open()) {
  //     std::cerr << "Error: Cannot open output file: " << centroids_file << std::endl;
  //     delete[] centroids;
  //     return;
  // }

  // uint32_t dimensions_centroids = d;
  // centroids_writer.write(reinterpret_cast<const char*>(&num_medoids), sizeof(uint32_t));
  // centroids_writer.write(reinterpret_cast<const char*>(&dimensions_centroids), sizeof(uint32_t));
  // centroids_writer.write(reinterpret_cast<const char*>(centroids), partition * d * sizeof(float));
  // centroids_writer.close();

  // // Free allocated memory
  // delete[] centroids;

  // std::cout << "index_centroids.bin saved successfully." << std::endl;
  

  // uint64_t nnodes = Points.size();
  // uint64_t ndims = Points.get_dims();
  // size_t max_node_len = sizeof(pointType) * ndims + sizeof(int) + BP.max_degree() * sizeof(indexType);
  // std::cout << "nnode = " << nnodes << ", ndims = " << ndims << ", max_node_len = " << max_node_len << std::endl;

  // std::string new_file_name(out_file);
  // new_file_name += "_diskann_disk.index";
  // std::cout << "New file name: " << new_file_name << std::endl;

  // size_t SECTOR_SIZE = 4096;  // 4096 bytes
  // size_t NNODES_PER_SECTOR = SECTOR_SIZE / max_node_len;

  // // Metadata
  // uint32_t nr = 9;              // Metadata entries
  // uint32_t nc = 1;              // Fixed to 1
  // size_t medoid_id = 0;         // Default medoid
  // size_t num_frozen_points = 0; // Default 0
  // uint64_t file_frozen_id = 0;  // Default 0
  // uint64_t reorder_data = 0;    // Default 0
  // size_t total_sectors = (nnodes + NNODES_PER_SECTOR - 1) / NNODES_PER_SECTOR;
  // size_t index_file_size = SECTOR_SIZE * (total_sectors+1); // Metadata sector + node data sectors
  // // size_t index_file_size = SECTOR_SIZE * (1 + nnodes / NNODES_PER_SECTOR);

  // // File setup
  // std::ofstream writer(new_file_name, std::ios::binary | std::ios::out);
  // if (!writer.is_open()) {
  //     std::cerr << "Error opening file for writing: " << new_file_name << std::endl;
  //     return;
  // }

  // // Step 1: Write metadata sector
  // writer.write(reinterpret_cast<const char*>(&nr), sizeof(uint32_t));
  // writer.write(reinterpret_cast<const char*>(&nc), sizeof(uint32_t));
  // writer.write(reinterpret_cast<const char*>(&nnodes), sizeof(uint64_t));
  // writer.write(reinterpret_cast<const char*>(&ndims), sizeof(uint64_t));
  // writer.write(reinterpret_cast<const char*>(&medoid_id), sizeof(size_t));
  // writer.write(reinterpret_cast<const char*>(&max_node_len), sizeof(size_t));
  // writer.write(reinterpret_cast<const char*>(&NNODES_PER_SECTOR), sizeof(size_t));
  // writer.write(reinterpret_cast<const char*>(&num_frozen_points), sizeof(uint64_t));
  // writer.write(reinterpret_cast<const char*>(&file_frozen_id), sizeof(uint64_t));
  // writer.write(reinterpret_cast<const char*>(&reorder_data), sizeof(uint64_t));
  // writer.write(reinterpret_cast<const char*>(&index_file_size), sizeof(size_t));

  // // Pad remaining metadata sector to SECTOR_SIZE
  // size_t metadata_size = 80;
  // char padding[SECTOR_SIZE] = {0};
  // writer.write(padding, SECTOR_SIZE - metadata_size);

  // // Step 2: Write node data sectors
  // size_t node_idx = 0;
  // while (node_idx < nnodes) {
  //     char sector[SECTOR_SIZE] = {0};
  //     int sector_cnt = 0;

  //     for (size_t i = 0; i < NNODES_PER_SECTOR && node_idx < nnodes; i++, node_idx++) {
  //         sector_cnt++;
  //         // Write node data
  //         for (int d = 0; d < ndims; d++) {
  //             pointType val = Points[node_idx][d];
  //             writer.write(reinterpret_cast<const char*>(&val), sizeof(pointType));
  //         }

  //         // Write number of edges
  //         uint32_t num_edges = G[node_idx].size();
  //         writer.write(reinterpret_cast<const char*>(&num_edges), sizeof(uint32_t));

  //         // Write edge list
  //         for (int j = 0; j < num_edges; j++) {
  //             indexType edge = G[node_idx][j];
  //             writer.write(reinterpret_cast<const char*>(&edge), sizeof(indexType));
  //         }

  //         // Pad remaining space for the node in the sector
  //         size_t node_size = ndims * sizeof(pointType) + sizeof(uint32_t) + num_edges * sizeof(indexType);
  //         // size_t node_size = ndims + sizeof(uint32_t) + num_edges * sizeof(uint32_t);
  //         size_t node_padding = max_node_len - node_size;
  //         if (node_padding > 0) {
  //             char pad[node_padding] = {0};
  //             writer.write(pad, node_padding);
  //         }
  //     }

  //     // Pad remaining space in the sector to SECTOR_SIZE
  //     if (sector_cnt > 0) {
  //       size_t sector_padding = SECTOR_SIZE - (sector_cnt * max_node_len);
  //       if (sector_padding > 0) {
  //           writer.write(padding, sector_padding);
  //       }
  //     }
      
  // }

  // writer.close();
  // std::cout << "DiskANN file saved to: " << new_file_name << std::endl;
  

  std::string name = "Vamana";
  std::string params =
    "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  auto [avg_deg, max_deg] = graph_stats_(G);
  auto vv = BuildStats.visited_stats();
  std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1] << ", Max visited: " << vv[2]
            << std::endl;

  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();  

  // long build_num_distances = parlay::reduce(parlay::map(BuildStats.distances,
  //                                                       [] (auto x) {return (long) x;}));

  // Search
  std::cout << "Querying" << std::endl;
  groundTruth<uint> GT = groundTruth<uint>(gt_file);
  indexType start_point = 0;
  if (disk_search) {
    PointRange<pointType, Point> Query_Points = PointRange<pointType, Point>(query_file);
  } else {
    PointRange<pointType, Point> Points = PointRange<pointType, Point>(input_file);
    PointRange<pointType, Point> Query_Points = PointRange<pointType, Point>(query_file);
    std::cout << "Query points: " << Query_Points.size() << std::endl;

    if(Query_Points.size() != 0) {
      search_and_parse<Point, PointRange_, PointRange_, indexType>(G_, G, Points, Query_Points,
                                                                  Points, Query_Points, GT,
                                                                  res_file, k, false, start_point,
                                                                  BP.verbose);
    }
  }

}

template<typename Point, typename PointRange_, typename indexType>
void ANN(Graph_disk<indexType> &G, long k, BuildParams &BP,
         PointRange_ &Query_Points,
         groundTruth<indexType> GT, char *res_file,
         bool graph_built, PointRange_ &Points, char* outFile) {
  parlay::internal::timer t("ANN");

  bool verbose = BP.verbose;
  using findex = knn_index<PointRange_, indexType>;
  findex I(BP);
  indexType start_point;
  double idx_time;
  stats<unsigned int> BuildStats(G.size());
  if(graph_built){
    idx_time = 0;
    start_point = 0;
  } else{
    I.build_index(G, Points, BuildStats);
    start_point = I.get_start();
    idx_time = t.next_time();
  }
  std::cout << "start index = " << start_point << std::endl;

  std::string name = "Vamana";
  std::string params =
    "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  // auto [avg_deg, max_deg] = graph_stats_(G);
  auto od = parlay::delayed_seq<size_t>(
      G.size(), [&](size_t i) { return G[i].size(); });
  size_t j = parlay::max_element(od) - od.begin();
  int max_deg = od[j];
  size_t sum1 = parlay::reduce(od);
  double avg_deg = sum1 / ((double)G.size());
  auto vv = BuildStats.visited_stats();
  std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
            << std::endl;

  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();

  long build_num_distances = parlay::reduce(parlay::map(BuildStats.distances,
                                                        [] (auto x) {return (long) x;}));

  if(outFile != NULL) {
    G.save(outFile);
  }

  Graph<indexType> G_r; 
  G_r = Graph<indexType>(outFile);

  if(Query_Points.size() != 0) {
    search_and_parse<Point, PointRange_, PointRange_, indexType>(G_, G_r, Points, Query_Points,
                                                                Points, Query_Points, GT,
                                                                res_file, k, false, start_point,
                                                                verbose);
  }
}