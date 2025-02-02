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

#pragma once

#include <sys/mman.h>
#include <algorithm>
#include <iostream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "../bench/parse_command_line.h"
#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//tp_size must divide 64 evenly--no weird/large types!
long dim_round_up(long dim, long tp_size){
  long qt = (dim*tp_size)/64;
  long remainder = (dim*tp_size)%64;
  if(remainder == 0) return dim;
  else return ((qt+1)*64)/tp_size;
}

  
template<typename T_, class Point_>
struct PointRange{
  using T = T_;
  using Point = Point_;
  using parameters = typename Point::parameters;

  long dimension() const {return dims;}
  long aligned_dimension() const {return aligned_dims;}

  PointRange() : values(std::shared_ptr<T[]>(nullptr, std::free)) {n=0;}

  template <typename PR>
  PointRange(const PR& pr, const parameters& p) : params(p)  {
    n = pr.size();
    dims = pr.dimension();
    aligned_dims =  dim_round_up(dims, sizeof(T));
    long num_bytes = n*aligned_dims*sizeof(T);
    T* ptr = (T*) aligned_alloc(1l << 21, num_bytes);
    madvise(ptr, num_bytes, MADV_HUGEPAGE);
    values = std::shared_ptr<T[]>(ptr, std::free);
    T* vptr = values.get();
    parlay::parallel_for(0, n, [&] (long i) {
      Point::translate_point(vptr + i * aligned_dims, pr[i], params);});
  }

  template <typename PR>
  PointRange (PR& pr) : PointRange(pr, Point::generate_parameters(pr)) { }

  PointRange(char* filename) : values(std::shared_ptr<T[]>(nullptr, std::free)){
      if(filename == NULL) {
        n = 0;
        dims = 0;
        return;
      }
      std::ifstream reader(filename);
      assert(reader.is_open());

      //read num points and max degree
      unsigned int num_points;
      unsigned int d;
      reader.read((char*)(&num_points), sizeof(unsigned int));
      n = num_points;
      reader.read((char*)(&d), sizeof(unsigned int));
      dims = d;
      params = parameters(d);
      std::cout << "Detected " << num_points << " points with dimension " << d << std::endl;
      aligned_dims =  dim_round_up(dims, sizeof(T));
      if(aligned_dims != dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;
      long num_bytes = n*aligned_dims*sizeof(T);
      T* ptr = (T*) aligned_alloc(1l << 21, num_bytes);
      madvise(ptr, num_bytes, MADV_HUGEPAGE);
      values = std::shared_ptr<T[]>(ptr, std::free);
      size_t BLOCK_SIZE = 1000000;
      size_t index = 0;
      while(index < n){
          size_t floor = index;
          size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
          T* data_start = new T[(ceiling-floor)*dims];
          reader.read((char*)(data_start), sizeof(T)*(ceiling-floor)*dims);
          T* data_end = data_start + (ceiling-floor)*dims;
          parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);
          int data_bytes = dims*sizeof(T);
          parlay::parallel_for(floor, ceiling, [&] (size_t i){
            for (int j=0; j < dims; j++)
              values.get()[i * aligned_dims + j] = data[(i - floor) * dims + j];
            //std::memmove(values.get() + i*aligned_dims, data.begin() + (i-floor)*dims, data_bytes);
          });
          delete[] data_start;
          index = ceiling;
      }
  }

  PointRange(char* filename, size_t start, size_t end) : values(std::shared_ptr<T[]>(nullptr, std::free)) {
    if (filename == NULL || start >= end) {
        n = 0;
        dims = 0;
        return;
    }

    std::ifstream reader(filename, std::ios::binary);
    assert(reader.is_open());

    // read number of points and dimension
    unsigned int num_points;
    unsigned int d;
    reader.read((char*)(&num_points), sizeof(unsigned int));
    reader.read((char*)(&d), sizeof(unsigned int));
    n = end - start;
    dims = d;
    params = parameters(d);
    std::cout << "Detected " << num_points << " points with dimension " << d << std::endl;
    aligned_dims = dim_round_up(dims, sizeof(T));
    if (aligned_dims != dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;

    if (end > num_points) {
        end = num_points;  // Ensure the end does not exceed the number of points in the file
    }
    long num_bytes = n * aligned_dims * sizeof(T);
    T* ptr = (T*)aligned_alloc(1l << 21, num_bytes);
    madvise(ptr, num_bytes, MADV_HUGEPAGE);
    values = std::shared_ptr<T[]>(ptr, std::free);
    size_t BLOCK_SIZE = 1000000;
    size_t index = start;

    // Move file cursor to the starting point
    reader.seekg(sizeof(unsigned int) * 2 + start * dims * sizeof(T));
    while (index < end) {
        size_t floor = index;
        size_t ceiling = index + BLOCK_SIZE <= end ? index + BLOCK_SIZE : end;
        size_t block_size = ceiling - floor;

        T* data_start = new T[block_size * dims];
        reader.read((char*)(data_start), sizeof(T) * block_size * dims);
        T* data_end = data_start + block_size * dims;
        parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);

        int data_bytes = dims * sizeof(T);
        parlay::parallel_for(floor, ceiling, [&](size_t i) {
            for (int j = 0; j < dims; j++) {
                values.get()[(i - start) * aligned_dims + j] = data[(i - floor) * dims + j];
            }
        });

        delete[] data_start;
        index = ceiling;
    }
}

  size_t size() const { return n; }

  unsigned int get_dims() const { return dims; }
  
  Point operator [] (long i) const {
    if (i > n) {
      std::cout << "ERROR: point index out of range: " << i << " from range " << n << ", " << std::endl;
      abort();
    }
    return Point(values.get()+i*aligned_dims, i, params);
  }

  // // 파일의 일부분을 읽어와 PointRange 끝에 추가하는 append 메서드
  // void append(char* filename, size_t start, size_t end) {
  //   if (filename == NULL || start >= end) {
  //     return;  // 파일이 없거나 잘못된 범위인 경우 아무것도 하지 않음
  //   }

  //   std::ifstream reader(filename, std::ios::binary);
  //   assert(reader.is_open());

  //   // 파일에서 전체 포인트 개수와 차원을 읽음
  //   unsigned int num_points, d;
  //   reader.read((char*)(&num_points), sizeof(unsigned int));
  //   reader.read((char*)(&d), sizeof(unsigned int));

  //   if (end > num_points) {
  //     end = num_points;  // 파일의 끝을 초과하지 않도록 조정
  //   }
  //   if (dims != d) {
  //     throw std::invalid_argument("Dimensions do not match");
  //   }

  //   size_t new_points_count = end - start;
  //   size_t new_size = n + new_points_count;
  //   size_t new_num_bytes = new_size * aligned_dims * sizeof(T);

  //   // 기존 값을 포함해 새로 데이터를 할당
  //   std::shared_ptr<T[]> new_values((T*)aligned_alloc(1l << 21, new_num_bytes), std::free);
  //   madvise(new_values.get(), new_num_bytes, MADV_HUGEPAGE);

  //   // 기존 데이터 복사
  //   T* old_values_ptr = values.get();
  //   T* new_values_ptr = new_values.get();
  //   parlay::parallel_for(0, n, [&](size_t i) {
  //     for (size_t j = 0; j < aligned_dims; ++j) {
  //       new_values_ptr[i * aligned_dims + j] = old_values_ptr[i * aligned_dims + j];
  //     }
  //   });

  //   // 새 데이터를 읽어와 복사
  //   reader.seekg(sizeof(unsigned int) * 2 + start * dims * sizeof(T));
  //   size_t BLOCK_SIZE = 1000000;
  //   size_t index = start;
  //   size_t offset = n;  // 새 데이터 시작 위치

  //   while (index < end) {
  //     size_t floor = index;
  //     size_t ceiling = std::min(index + BLOCK_SIZE, end);
  //     size_t block_size = ceiling - floor;

  //     T* data_start = new T[block_size * dims];
  //     reader.read((char*)(data_start), sizeof(T) * block_size * dims);
  //     T* data_end = data_start + block_size * dims;
  //     parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);

  //     parlay::parallel_for(floor, ceiling, [&](size_t i) {
  //       size_t local_index = (i - floor) + offset;
  //       for (int j = 0; j < dims; ++j) {
  //         new_values_ptr[local_index * aligned_dims + j] = data[(i - floor) * dims + j];
  //       }
  //     });

  //     delete[] data_start;
  //     index = ceiling;
  //     offset += block_size;
  //   }

  //   // 새로운 값을 할당하고 개수 업데이트
  //   values = std::move(new_values);
  //   n = new_size;
  // }

  parameters params;

private:
  std::shared_ptr<T[]> values;
  unsigned int dims;
  unsigned int aligned_dims;
  size_t n;
};
