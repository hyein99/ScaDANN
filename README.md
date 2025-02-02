# ScaDANN: Scalable Disk-based Graph Index for ANN

ScaDANN is a scalable, disk-based approximate nearest neighbor (ANN) search library designed for billion-scale datasets under memory-constrained environments. Built upon the **ParlayANN** framework and leveraging **ParlayLib**, ScaDANN introduces novel techniques such as **overlapping block-level insertion** and **grid block merge** to optimize both index construction and search performance. 

This repository provides an efficient and scalable implementation of **ScaDANN**, enabling high-performance ANN search while reducing memory overhead and I/O costs.

## Features

- **Efficient Disk-Based ANN Search**: Optimized for large-scale datasets beyond in-memory constraints.
- **Overlapping Block-Level Insertion**: Incrementally constructs graphs by reusing blocks, reducing I/O overhead.
- **Grid Block Merge**: Merges partitioned graphs while preserving distance information to improve search performance.
- **Parallel Processing with ParlayLib**: Utilizes parallel primitives for optimized graph construction and search.

## Installation

To install ScaDANN, first **clone the repository** and initialize the ParlayLib submodule:

```bash
git clone https://github.com/hyein99/ScaDANN.git
cd ScaDANN
git submodule init
git submodule update
```

## Usage
```bash
cd algorithms/vamana_disk
make
```


### Building an Index

To construct a **ScaDANN** index, use the following command:

```bash
./neighbors \
    -R 64 \
    -L 100 \
    -graph_outfile <graph_path> \
    -base_path <base_file_path>  \
    -query_path <query_file_path> \
    -gt_path <groundtruth_file_path> \
    -data_type <data_type> \
    -dist_func <distance_function> \
    -disk \
    -partition <partition_number> \
    -partition_batch
```

### Parameter	Description
* `R`: Maximum number of neighbors per node (default: 64)
* `L`: Search list size for graph construction (default: 100)
* `graph_outfile`: Path to store the constructed graph index
* `base_path`: Path to the input dataset file
* `query_path`:	Path to the query dataset file
* `gt_path`: Path to the ground truth nearest neighbors file
* `data_type`: Data format (float, int, or uint8)
* `dist_func`: Distance metric to use (Euclidean, Cosine, etc.)
* `disk`:	Enables disk-based index construction
* `partition`: Number of partitions for disk-based processing
* `partition_batch`: Enables batch-based partitioning