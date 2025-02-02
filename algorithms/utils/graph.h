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
#include "point_range.h"

template<typename indexType>
struct edgeRange{

  size_t size() const {return edges[0];}

  indexType id() const {return id_;}

  edgeRange() : edges(parlay::make_slice<indexType*, indexType*>(nullptr, nullptr)) {}

  edgeRange(indexType* start, indexType* end, indexType id)
    : edges(parlay::make_slice<indexType*, indexType*>(start,end)), id_(id) {
    maxDeg = edges.size() - 1;
  }

  indexType operator [] (indexType j) const {
    if (j > edges[0]) {
      std::cout << "ERROR: index exceeds degree while accessing neighbors" << std::endl;
      abort();
    } else return edges[j+1];
  }

  void append_neighbor(indexType nbh){
    if (edges[0] == maxDeg) {
      std::cout << "ERROR in append_neighbor: cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    } else {
      edges[edges[0]+1] = nbh;
      edges[0] += 1;
    }
  }

  template<typename rangeType>
  void update_neighbors(const rangeType& r){
    if (r.size() > maxDeg) {
      std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    }
    edges[0] = r.size();
    for (int i = 0; i < r.size(); i++) {
      edges[i+1] = r[i];
    }
  }

  template<typename rangeType>
  void append_neighbors(const rangeType& r){
    if (r.size() + edges[0] > maxDeg) {
      std::cout << "ERROR in append_neighbors for point " << id_
                << ": cannot exceed max degree " << maxDeg << std::endl;
      std::cout << edges[0] << std::endl;
      std::cout << r.size() << std::endl;
      abort();
    }
    for (int i = 0; i < r.size(); i++) {
      edges[edges[0] + i + 1] = r[i];
    }
    edges[0] += r.size();
  }

  void clear_neighbors(){
    edges[0] = 0;
  }

  void prefetch(){
    int l = ((edges[0] + 1) * sizeof(indexType))/64;
    for (int i = 0; i < l; i++)
      __builtin_prefetch((char*) edges.begin() + i *  64);
  }

  template<typename F>
  void sort(F&& less){
    std::sort(edges.begin() + 1, edges.begin() + 1 + edges[0], less);}

  indexType* begin(){return edges.begin() + 1;}

  // indexType* end(){return edges.end() + 1 + edges[0];}

private:
  parlay::slice<indexType*, indexType*> edges;
  long maxDeg;
  indexType id_;
};

template<typename indexType>
struct Graph{
  long max_degree() const {return maxDeg;}
  size_t size() const {return n;}

  Graph(){}

  void allocate_graph(long maxDeg, size_t n) {
    long cnt = n * (maxDeg + 1);
    long num_bytes = cnt * sizeof(indexType);
    indexType* ptr = (indexType*) aligned_alloc(1l << 21, num_bytes);
    madvise(ptr, num_bytes, MADV_HUGEPAGE);
    parlay::parallel_for(0, cnt, [&] (long i) {ptr[i] = 0;});
    graph = std::shared_ptr<indexType[]>(ptr, std::free);
  }

  Graph(long maxDeg, size_t n) : maxDeg(maxDeg), n(n) {
    allocate_graph(maxDeg, n);
  }

  Graph(char* gFile){
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
        indexType* edges_start = new indexType[total_size_to_read];
        reader.read((char*) (edges_start), sizeof(indexType) * total_size_to_read);
        indexType* edges_end = edges_start + total_size_to_read;
        parlay::slice<indexType*, indexType*> edges =
          parlay::make_slice(edges_start, edges_end);
        indexType* gr = graph.get();
        parlay::parallel_for(g_floor, g_ceiling, [&] (size_t i){
          gr[i * (maxDeg + 1)] = degrees[i];
          for(size_t j = 0; j < degrees[i]; j++){
            gr[i * (maxDeg + 1) + 1 + j] = edges[offsets[i] - total_size_read + j];
          }
        });
      total_size_read += total_size_to_read;
      index = g_ceiling;
      delete[] edges_start;
    }
    delete[] degrees_start;
  }


  Graph(char* gFile, indexType total_num_points, indexType total_max_deg, int partition) {
    n = total_num_points;
    maxDeg = total_max_deg;
    allocate_graph(total_max_deg, total_num_points);
    std::string graphfile(gFile);

    size_t global_offset = 0;

    for (int p = 0; p < partition; p++) {
      std::ifstream reader(const_cast<char*>((graphfile + "_sub_" + std::to_string(p)).c_str()), 
                                              std::ios::binary);
      assert(reader.is_open());

      // num_points와 max_deg 읽기
      indexType num_points;
      indexType max_deg;
      reader.read((char*)(&num_points), sizeof(indexType));
      reader.read((char*)(&max_deg), sizeof(indexType));
      std::cout << "Detected " << num_points
                << " points with max degree " << max_deg << std::endl;

      // degree 읽기
      indexType* degrees_start = new indexType[n];
      reader.read((char*) (degrees_start), sizeof(indexType) * num_points);
      indexType* degrees_end = degrees_start + num_points;
      parlay::slice<indexType*, indexType*> degrees0 =
        parlay::make_slice(degrees_start, degrees_end);
      auto degrees = parlay::tabulate(degrees0.size(), [&] (size_t i){
        return static_cast<size_t>(degrees0[i]);});
      auto [offsets, total] = parlay::scan(degrees);
      std::cout << "Total edges read from partition " << p << ": " << total << std::endl;
      offsets.push_back(total);

      // 1000000개씩 읽기
      size_t BLOCK_SIZE = 1000000;
      size_t index = 0;
      size_t total_size_read = 0;

      // 전체 그래프에서의 현재 파티션의 시작 지점(오프셋) 계산
      size_t partition_start_idx = global_offset;
      while(index < num_points) {
        size_t g_floor = index;
        size_t g_ceiling = g_floor + BLOCK_SIZE <= num_points ? g_floor + BLOCK_SIZE : num_points;
        size_t total_size_to_read = offsets[g_ceiling] - offsets[g_floor];
        
        // neighbor와 distance를 포함한 이웃 정보 읽기
        std::pair<indexType, float>* neighbor_distance_pairs = new std::pair<indexType, float>[total_size_to_read];
        reader.read(reinterpret_cast<char*>(neighbor_distance_pairs), sizeof(std::pair<indexType, float>) * total_size_to_read);

        // auto* gr = reinterpret_cast<std::pair<indexType, float>*>(graph.get());
        auto* gr = graph.get();

        parlay::parallel_for(g_floor, g_ceiling, [&](size_t i) {
            // 파티션 내 인덱스 `i`를 전체 그래프 인덱스에 맞게 변환
            size_t global_idx = partition_start_idx + i;

            gr[global_idx * (total_max_deg + 1)] = degrees[i];  // degree 저장

            // neighbor 정보 저장 (distance는 무시하고 neighbor ID만 저장)
            for (size_t j = 0; j < degrees[i]; j++) {
                gr[global_idx * (total_max_deg + 1) + 1 + j] = neighbor_distance_pairs[offsets[i] - total_size_read + j].first;
            }
        });

        total_size_read += total_size_to_read;
        index = g_ceiling;

        delete[] neighbor_distance_pairs;
      }
      
      delete[] degrees_start;

      global_offset += num_points;
    
    }
    
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
          return (*this)[i + floor][j];});
      });
      parlay::sequence<indexType> data = parlay::flatten(edge_data);
      writer.write((char*)data.begin(), data.size() * sizeof(indexType));
      index = ceiling;
    }
    writer.close();
  }

  void save_vamana(char* oFile, uint32_t start_point = 0) {
    std::cout << "Writing graph in Vamana format with " << n
              << " points and max degree " << maxDeg
              << std::endl;

    std::string new_file_name(oFile);
    new_file_name += "_vamana";
    std::cout << "New file name: " << new_file_name << std::endl;
    std::ofstream writer(new_file_name, std::ios::binary | std::ios::out);

    // Step 1: Calculate file size
    size_t file_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t); // Metadata
    size_t edge_data_size = 0;

    parlay::sequence<uint32_t> degrees(n);
    for (size_t i = 0; i < n; i++) {
      degrees[i] = (*this)[i].size();
      file_size += sizeof(uint32_t); // Degree of node
      edge_data_size += degrees[i];
    }
    file_size += edge_data_size * sizeof(uint32_t); // All edges

    // Step 2: Write metadata
    size_t num_frozen_points = 0;
    writer.write(reinterpret_cast<char*>(&file_size), sizeof(size_t));        // File size
    writer.write(reinterpret_cast<char*>(&maxDeg), sizeof(uint32_t));         // Max degree
    writer.write(reinterpret_cast<char*>(&start_point), sizeof(uint32_t));    // Start point
    writer.write(reinterpret_cast<char*>(&num_frozen_points), sizeof(size_t));// Number of frozen points

    // Step 3: Write node data (degree + edges)
    for (size_t i = 0; i < n; i++) {
      uint32_t degree = degrees[i];
      writer.write(reinterpret_cast<char*>(&degree), sizeof(uint32_t)); // Row degree

      for (size_t j = 0; j < degree; j++) {
        uint32_t edge = (*this)[i][j];
        writer.write(reinterpret_cast<char*>(&edge), sizeof(uint32_t)); // Edges
      }
    }

    writer.close();
    std::cout << "Graph saved to " << new_file_name << " successfully." << std::endl;
  }

  // void save_diskann(const char* oFile, PointRange &Points, uint64_t nnodes, uint64_t dims, size_t max_node_len) {
  //     std::string new_file_name(oFile);
  //     new_file_name += "_diskann";
  //     std::cout << "New file name: " << new_file_name << std::endl;

  //     size_t SECTOR_SIZE = 4096;  // 4096 bytes
  //     size_t NNODES_PER_SECTOR = 10;

  //     // Metadata
  //     uint32_t nr = 9;              // Metadata entries
  //     uint32_t nc = 1;              // Fixed to 1
  //     uint64_t nnodes = Points.size(); // Number of nodes
  //     size_t medoid_id = 0;         // Default medoid
  //     size_t num_frozen_points = 0; // Default 0
  //     uint64_t file_frozen_id = 0;  // Default 0
  //     uint64_t reorder_data = 0;    // Default 0
  //     size_t index_file_size = SECTOR_SIZE * (1 + nnodes / NNODES_PER_SECTOR);

  //     // File setup
  //     std::ofstream writer(new_file_name, std::ios::binary | std::ios::out);
  //     if (!writer.is_open()) {
  //         std::cerr << "Error opening file for writing: " << new_file_name << std::endl;
  //         return;
  //     }

  //     // Step 1: Write metadata sector
  //     writer.write(reinterpret_cast<const char*>(&nr), sizeof(uint32_t));
  //     writer.write(reinterpret_cast<const char*>(&nc), sizeof(uint32_t));
  //     writer.write(reinterpret_cast<const char*>(&nnodes), sizeof(uint64_t));
  //     writer.write(reinterpret_cast<const char*>(&dims), sizeof(uint64_t));
  //     writer.write(reinterpret_cast<const char*>(&medoid_id), sizeof(size_t));
  //     writer.write(reinterpret_cast<const char*>(&max_node_len), sizeof(size_t));
  //     writer.write(reinterpret_cast<const char*>(&NNODES_PER_SECTOR), sizeof(size_t));
  //     writer.write(reinterpret_cast<const char*>(&num_frozen_points), sizeof(uint64_t));
  //     writer.write(reinterpret_cast<const char*>(&file_frozen_id), sizeof(uint64_t));
  //     writer.write(reinterpret_cast<const char*>(&reorder_data), sizeof(uint64_t));
  //     writer.write(reinterpret_cast<const char*>(&index_file_size), sizeof(size_t));

  //     // Pad remaining metadata sector to SECTOR_SIZE
  //     size_t metadata_size = 9 * sizeof(uint32_t) + 4 * sizeof(uint64_t) + 3 * sizeof(size_t);
  //     char padding[SECTOR_SIZE] = {0};
  //     writer.write(padding, SECTOR_SIZE - metadata_size);

  //     // Step 2: Write node data sectors
  //     size_t node_idx = 0;
  //     while (node_idx < nnodes) {
  //         char sector[SECTOR_SIZE] = {0};
  //         size_t sector_offset = 0;

  //         for (size_t i = 0; i < NNODES_PER_SECTOR && node_idx < nnodes; i++, node_idx++) {
  //             // Write node data
  //             writer.write(reinterpret_cast<const char*>(Points[node_idx].data()), dims);

  //             // Write number of edges
  //             uint32_t num_edges = (*this)[node_idx].size();
  //             writer.write(reinterpret_cast<const char*>(&num_edges), sizeof(uint32_t));

  //             // Write edge list
  //             for (uint32_t edge : (*this)[node_idx]) {
  //                 writer.write(reinterpret_cast<const char*>(&edge), sizeof(uint32_t));
  //             }

  //             // Pad remaining space for the node in the sector
  //             size_t node_size = dims + sizeof(uint32_t) + num_edges * sizeof(uint32_t);
  //             size_t node_padding = max_node_len - node_size;
  //             if (node_padding > 0) {
  //                 char pad[node_padding] = {0};
  //                 writer.write(pad, node_padding);
  //             }
  //         }

  //         // Pad remaining space in the sector to SECTOR_SIZE
  //         size_t sector_padding = SECTOR_SIZE - (NNODES_PER_SECTOR * max_node_len);
  //         if (sector_padding > 0) {
  //             writer.write(padding, sector_padding);
  //         }
  //     }

  //     writer.close();
  //     std::cout << "DiskANN file saved to: " << new_file_name << std::endl;
  // }
  

  edgeRange<indexType> operator [] (indexType i) {
    if (i > n) {
      std::cout << "ERROR: graph index out of range: " << i << std::endl;
      abort();
    }
    return edgeRange<indexType>(graph.get() + i * (maxDeg + 1),
                                graph.get() + (i + 1) * (maxDeg + 1),
                                i);
  }

  ~Graph(){}

private:
  size_t n;
  long maxDeg;
  std::shared_ptr<indexType[]> graph;
};
