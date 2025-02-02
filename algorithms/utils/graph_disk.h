// This code is part of the Parlay Project
// Copyright (c) 2024 Guy Blelloch, Magdalen Dobson and the Parlay team
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

#pragma once

#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"

#include "../bench/parse_command_line.h"
#include "types.h"
#include "graph.h"

template<typename indexType>
struct edgeRange_disk{

  size_t size() const {return edges[0].first;}

  indexType id() const {return id_;}

  // edgeRange_disk() : edges(parlay::make_slice<indexType*, indexType*>(nullptr, nullptr)) {}

  // edgeRange_disk(indexType* start, indexType* end, indexType id)
  //   : edges(parlay::make_slice<indexType*, indexType*>(start,end)), id_(id) {
  //   maxDeg = edges.size() - 1;
  // }

  edgeRange_disk() : edges(parlay::make_slice<std::pair<indexType, float>*, std::pair<indexType, float>*>(nullptr, nullptr)) {}

  edgeRange_disk(std::pair<indexType, float>* start, std::pair<indexType, float>* end, indexType id)
    : edges(parlay::make_slice<std::pair<indexType, float>*, std::pair<indexType, float>*>(start, end)), id_(id) {
    maxDeg = edges.size() - 1;
  }

  // indexType operator [] (indexType j) const {
  //   if (j > edges[0]) {
  //     std::cout << "ERROR: index exceeds degree while accessing neighbors" << std::endl;
  //     abort();
  //   } else return edges[j+1];
  // }

  std::pair<indexType, float> operator [] (indexType j) const {
    if (j >= edges[0].first) {
      std::cout << "ERROR: index exceeds degree while accessing neighbors" << std::endl;
      abort();
    }
    return edges[j+1];  // neighbor 정보는 첫 번째 pair 이후에 저장됨
  }

  // void append_neighbor(indexType nbh){
  //   if (edges[0].first == maxDeg) {
  //     std::cout << "ERROR in append_neighbor: cannot exceed max degree "
  //               << maxDeg << std::endl;
  //     abort();
  //   } else {
  //     edges[edges[0]+1] = nbh;
  //     edges[0] += 1;
  //   }
  // }

  template<typename rangeType>
  void update_neighbors(const rangeType& r){
    if (r.size() > maxDeg) {
      std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    }
    edges[0].first = r.size();
    for (int i = 0; i < r.size(); i++) {
      edges[i+1] = r[i];
    }
  }

  template<typename rangeType>
  void update_neighbors_global(const rangeType& r, int offset){
    if (r.size() > maxDeg) {
      std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    }
    edges[0].first = r.size();
    for (int i = 0; i < r.size(); i++) {
      edges[i+1] = std::make_pair(r[i].first + offset, r[i].second);
    }
  }

  // template<typename rangeType>
  // void append_neighbors(const rangeType& r){
  //   if (r.size() + edges[0] > maxDeg) {
  //     std::cout << "ERROR in append_neighbors for point " << id_
  //               << ": cannot exceed max degree " << maxDeg << std::endl;
  //     std::cout << edges[0] << std::endl;
  //     std::cout << r.size() << std::endl;
  //     abort();
  //   }
  //   for (int i = 0; i < r.size(); i++) {
  //     edges[edges[0] + i + 1] = r[i];
  //   }
  //   edges[0] += r.size();
  // }

  // void clear_neighbors(){
  //   edges[0] = 0;
  // }

  void prefetch(){
    int l = ((edges[0].first + 1) * sizeof(indexType))/64;
    for (int i = 0; i < l; i++)
      __builtin_prefetch((char*) edges.begin() + i *  64);
  }

  template<typename F>
  void sort(F&& less){
    std::sort(edges.begin() + 1, edges.begin() + 1 + edges[0].first, less);}

  indexType* begin(){return edges.begin() + 1;}

  indexType* end(){return edges.end() + 1 + edges[0];}

private:
  // parlay::slice<indexType*, indexType*> edges;
  parlay::slice<std::pair<indexType, float>*, std::pair<indexType, float>*> edges;
  long maxDeg;
  indexType id_;
};


template<typename indexType>
struct Graph_disk{
  long max_degree() const {return maxDeg;}
  size_t size() const {return n;}

  Graph_disk(){}

  void allocate_graph(long maxDeg, size_t n) {
    long cnt = n * (maxDeg + 1);
    long num_bytes = cnt * sizeof(std::pair<indexType, float>);
    void* ptr = aligned_alloc(1l << 21, num_bytes);
    madvise(ptr, num_bytes, MADV_HUGEPAGE);
    parlay::parallel_for(0, cnt, [&] (long i) {
      auto* pair_ptr = reinterpret_cast<std::pair<indexType, float>*>(ptr);
      pair_ptr[i] = std::make_pair(0, -1.0f);  // Initialize both neighbor and distance to 0 and -1.0f
    });
    // Store the pointer to the allocated graph memory (neighbors + distances)
    graph = std::shared_ptr<std::pair<indexType, float>[]>(reinterpret_cast<std::pair<indexType, float>*>(ptr), std::free);
  }

  Graph_disk(long maxDeg, size_t n) : maxDeg(maxDeg), n(n) {
    allocate_graph(maxDeg, n);
  }

  Graph_disk(char* gFile) {
    std::ifstream reader(gFile);
    assert(reader.is_open());

    //read num points and max degree
    indexType num_points;
    indexType max_deg;
    reader.read((char*)(&num_points), sizeof(indexType));
    n = num_points;
    reader.read((char*)(&max_deg), sizeof(indexType));
    maxDeg = max_deg;
    std::cout << "Detected " << num_points
              << " points with max degree " << max_deg << std::endl;

    //read degrees and perform scan to find offsets
    indexType* degrees_start = new indexType[n];
    reader.read((char*) (degrees_start), sizeof(indexType) * n);
    indexType* degrees_end = degrees_start + n;
    parlay::slice<indexType*, indexType*> degrees0 =
      parlay::make_slice(degrees_start, degrees_end);
    auto degrees = parlay::tabulate(degrees0.size(), [&] (size_t i){
      return static_cast<size_t>(degrees0[i]);});
    auto [offsets, total] = parlay::scan(degrees);
    std::cout << "Total edges read from file: " << total << std::endl;
    offsets.push_back(total);

    allocate_graph(max_deg, n);

    //write 1000000 vertices at a time
    size_t BLOCK_SIZE = 1000000;
    size_t index = 0;
    size_t total_size_read = 0;
    while(index < n){
      size_t g_floor = index;
      size_t g_ceiling = g_floor + BLOCK_SIZE <= n ? g_floor + BLOCK_SIZE : n;
      size_t total_size_to_read = offsets[g_ceiling] - offsets[g_floor];
      
      // indexType* neighbors_start = new indexType[total_size_to_read];
      // reader.read(reinterpret_cast<char*>(neighbors_start), sizeof(indexType) * total_size_to_read);
      // float* distances_start = new float[total_size_to_read];
      // reader.read(reinterpret_cast<char*>(distances_start), sizeof(float) * total_size_to_read);

      // 이웃 ID와 거리를 pair로 읽어오기
      std::pair<indexType, float>* neighbor_distance_pairs = new std::pair<indexType, float>[total_size_to_read];
      reader.read(reinterpret_cast<char*>(neighbor_distance_pairs), sizeof(std::pair<indexType, float>) * total_size_to_read);

      auto* gr = reinterpret_cast<std::pair<indexType, float>*>(graph.get());

      parlay::parallel_for(g_floor, g_ceiling, [&](size_t i) {
          gr[i * (maxDeg + 1)] = std::make_pair(degrees[i], -1.0f);

          // 이웃 노드와 거리를 pair로 저장
          for (size_t j = 0; j < degrees[i]; j++) {
              gr[i * (maxDeg + 1) + 1 + j] = neighbor_distance_pairs[offsets[i] - total_size_read + j];
          }
      });

      total_size_read += total_size_to_read;
      index = g_ceiling;

      delete[] neighbor_distance_pairs;
      
    }
    delete[] degrees_start;
  }

  void save(char* oFile) {
    std::cout << "Writing graph with " << n
              << " points and max degree " << maxDeg
              << std::endl;
    parlay::sequence<indexType> preamble =
      {static_cast<indexType>(n), static_cast<indexType>(maxDeg)};
    parlay::sequence<indexType> sizes = parlay::tabulate(n, [&] (size_t i){
      return static_cast<indexType>((*this)[i].size());});
    std::ofstream writer;
    writer.open(oFile, std::ios::binary | std::ios::out);
    writer.write((char*) preamble.begin(), 2 * sizeof(indexType));
    writer.write((char*) sizes.begin(), sizes.size() * sizeof(indexType));
    size_t BLOCK_SIZE = 1000000;
    size_t index = 0;
    while(index < n){
      size_t floor = index;
      size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
      auto edge_data = parlay::tabulate(ceiling - floor, [&] (size_t i){
        return parlay::tabulate(sizes[i + floor], [&] (size_t j){
          return (*this)[i + floor][j].first;});
      });
      parlay::sequence<indexType> data = parlay::flatten(edge_data);
      writer.write((char*)data.begin(), data.size() * sizeof(indexType));
      index = ceiling;
    }
    writer.close();
  }

  void save_subgraph(char* oFile) {
    std::cout << "Writing graph with " << n
              << " points and max degree " << maxDeg
              << std::endl;
    parlay::sequence<indexType> preamble =
      {static_cast<indexType>(n), static_cast<indexType>(maxDeg)};
    parlay::sequence<indexType> sizes = parlay::tabulate(n, [&] (size_t i){
      return static_cast<indexType>((*this)[i].size());});
    std::ofstream writer;
    writer.open(oFile, std::ios::binary | std::ios::out);
    if (!writer.is_open()) {
        std::cerr << "ERROR: Failed to open file for writing." << std::endl;
        return;
    }
    writer.write((char*) preamble.begin(), 2 * sizeof(indexType));
    writer.write((char*) sizes.begin(), sizes.size() * sizeof(indexType));
    size_t BLOCK_SIZE = 1000000;
    size_t index = 0;
    while (index < n) {
        size_t floor = index;
        size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
        
        // 이웃 노드 ID와 거리 정보를 저장할 구조 생성
        auto edge_data = parlay::tabulate(ceiling - floor, [&] (size_t i){
            return parlay::tabulate(sizes[i + floor], [&] (size_t j){
                if (j >= (*this)[i + floor].size()) {
                    std::cerr << "ERROR: Degree index exceeds bounds at node " << i + floor << std::endl;
                    return std::make_pair(static_cast<indexType>(-1), -1.0f);
                }
                return std::make_pair((*this)[i + floor][j].first, (*this)[i + floor][j].second);
                // return std::make_pair((*this)[i + floor][j+1].first, (*this)[i + floor][j+1].second);
            });
        });
        parlay::sequence<std::pair<indexType, float>> data = parlay::flatten(edge_data);
        writer.write((char*)data.begin(), data.size() * sizeof(std::pair<indexType, float>));
        
        index = ceiling;
    }

    writer.close();
    if (!writer) {
        std::cerr << "ERROR: Failed to close file after writing." << std::endl;
    } else {
        std::cout << "Graph saved successfully to " << oFile << std::endl;
    }
}

  edgeRange_disk<indexType> operator [] (indexType i) {
    if (i > n) {
      std::cout << "ERROR: graph index out of range: " << i << std::endl;
      abort();
    }

    return edgeRange_disk<indexType>(graph.get() + i * (maxDeg + 1),
                                graph.get() + (i + 1) * (maxDeg + 1),
                                i);
  }

  ~Graph_disk(){}

private:
  size_t n;
  long maxDeg;
  std::shared_ptr<std::pair<indexType, float>[]> graph;
};
