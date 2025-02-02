#! /bin/bash

# sift-128-euclidean
# ./neighbors \
#     -R 32 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/sift_R32_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/base.fbin  \
#     -query_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/query.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/groundtruth.neighbors.ibin \
#     -data_type float \
#     -dist_func Euclidian > sift_R32_L100_alpha12.log 2>&1

# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/sift_1m \
#     -base_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/base.fbin  \
#     -query_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/query.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/groundtruth.neighbors.ibin \
#     -data_type float \
#     -dist_func Euclidian > sift_1m.log 2>&1



# # bigann-10M
# ./neighbors \
#     -R 32 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/bigann10M_R32_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > bigann10M_R32_L100_alpha12.log 2>&1

# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/sift_10m \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > sift_10m.log 2>&1


# # bigann-100M
# ./neighbors \
#     -R 32 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/bigann100M_R32_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-100M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-100M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-100M/bigann-100M.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > bigann100M_R32_L100_alpha12.log 2>&1

# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/sift_100m \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-100M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-100M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-100M/bigann-100M.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > sift_100m.log 2>&1


# # bigann-1B
# ./neighbors \
#     -R 32 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/bigann1B_R32_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/bigann/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann/GT.public.1B.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > bigann1B_R32_L100_alpha12.log 2>&1

# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/sift_1b \
#     -base_path /mnt/disk1/hyein99/datasets/bigann/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann/GT.public.1B.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > sift_1b.log 2>&1


# # deep1B
# ./neighbors \
#     -R 32 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/deep1B_R32_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/deep/base.1B.fbin \
#     -query_path /mnt/disk1/hyein99/datasets/deep/query.public.10K.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/deep/deep_new_groundtruth.public.10K.bin \
#     -data_type float \
#     -dist_func Euclidian > deep1B_R32_L100_alpha12.log 2>&1

# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/deep1B_R64_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/deep/base.1B.fbin \
#     -query_path /mnt/disk1/hyein99/datasets/deep/query.public.10K.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/deep/deep_new_groundtruth.public.10K.bin \
#     -data_type float \
#     -dist_func Euclidian > deep1B_R64_L100_alpha12.log 2>&1

# t2i 10m
# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 \
#     -num_passes 2 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/t2i1b_R64_L100_alpha12_two \
#     -base_path /mnt/disk1/hyein99/datasets/text2image1B/base.1B.fbin \
#     -query_path /mnt/disk1/hyein99/datasets/text2image1B/query.public.100K.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/text2image1B/t2i_new_groundtruth.public.100K.bin \
#     -data_type float \
#     -dist_func mips > t2i1b_R64_L100_alpha12_two.log 2>&1


# # memory usage
# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/sift_R64_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/base.fbin  \
#     -query_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/query.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/groundtruth.neighbors.ibin \
#     -data_type float \
#     -dist_func Euclidian > sift_R64_L100_alpha12_memory_usage.log 2>&1

# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/bigann10M_R64_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > bigann10M_R64_L100_alpha12_memory_usage.log 2>&1

# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/bigann100M_R64_L100_alpha12_usg \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-100M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-100M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-100M/bigann-100M.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > bigann100M_R64_L100_alpha12_memory_usage.log 2>&1

# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/bigann1B_R64_L100_alpha12 \
#     -base_path /mnt/disk1/hyein99/datasets/bigann/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann/GT.public.1B.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > bigann1B_R64_L100_alpha12_memory_usage.log 2>&1

./neighbors \
    -R 64 \
    -L 100 \
    -alpha 1.2 two_pass 0 \
    -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/msturing1B \
    -base_path /mnt/disk1/hyein99/datasets/MSTuringANNS/base1b.fbin \
    -query_path /mnt/disk1/hyein99/datasets/MSTuringANNS/query100K.fbin \
    -gt_path /mnt/disk1/hyein99/datasets/MSTuringANNS/query_gt100.bin \
    -data_type float \
    -dist_func Euclidian > msturing1b.log 2>&1\


# # cuts test
# ./neighbors \
    # -R 64 \
    # -L 100 \
    # -alpha 1.2 two_pass 0 \
    # -graph_outfile /mnt/disk2/hyein99/ParlayANN_experiments/sift_R64_L100_alpha12_cut100 \
    # -base_path /mnt/disk2/hyein99/datasets/sift-128-euclidean/base.fbin  \
    # -query_path /mnt/disk2/hyein99/datasets/sift-128-euclidean/query.fbin \
    # -gt_path /mnt/disk2/hyein99/datasets/sift-128-euclidean/groundtruth.neighbors.ibin \
    # -data_type float \
    # -dist_func Euclidian > sift_R64_L100_alpha12_cut100.log 2>&1


# =================================================================================================
# disk test
# sift-1m
# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_1m_merge50_part16/graph \
#     -base_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/base.fbin  \
#     -query_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/query.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/groundtruth.neighbors.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_1m_merge50_part16.log 2>&1

# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_1m_merge75_part16/graph \
#     -base_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/base.fbin  \
#     -query_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/query.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/groundtruth.neighbors.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_1m_merge75_part16.log 2>&1

# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_1m_merge100_part16/graph \
#     -base_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/base.fbin  \
#     -query_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/query.fbin \
#     -gt_path /mnt/disk1/hyein99/datasets/sift-128-euclidean/groundtruth.neighbors.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_1m_merge100_part16.log 2>&1


# sift-10m
# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_10m_merge50_part16/graph \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_10m_merge50_part16.log 2>&1

# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_10m_merge75_part16/graph \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_10m_merge75_part16.log 2>&1

# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_10m_merge100_part16/graph \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_10m_merge100_part16.log 2>&1

# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_10m_merge50_part32/graph \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_10m_merge50_part32.log 2>&1

# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_10m_merge75_part32/graph \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_10m_merge75_part32.log 2>&1

# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -graph_outfile /mnt/disk1/hyein99/BigDiskANN_experiments/sift_10m_merge100_part32/graph \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type float \
#     -dist_func Euclidian \
#     -disk \
#     -partition 16 \
#     -partition_batch > log/sift_10m_merge100_part32.log 2>&1


# ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_path /mnt/disk1/hyein99/BigDiskANN_experiments/sift100m_part4_max0.1_merge1/graph \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-100M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-100M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-100M/bigann-100M.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian


# make && ./neighbors \
#     -R 64 \
#     -L 100 \
#     -alpha 1.2 two_pass 0 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/sift_10m_tmp \
#     -base_path /mnt/disk1/hyein99/datasets/bigann-10M/base.1B.u8bin \
#     -query_path /mnt/disk1/hyein99/datasets/bigann-10M/query.public.10K.u8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/bigann-10M/bigann-10M.ibin \
#     -data_type uint8 \
#     -dist_func Euclidian > example.log 2>&1

# make && ./neighbors \
#     -R 100 \
#     -L 500 \
#     -alpha 1.2 \
#     -num_passes 2 \
#     -file_type bin \
#     -graph_outfile /mnt/disk1/hyein99/ParlayANN_experiments/spacev_1b \
#     -base_path /mnt/disk1/hyein99/datasets/MSSPACEV1B/spacev1b_base.i8bin \
#     -query_path /mnt/disk1/hyein99/datasets/MSSPACEV1B/query.i8bin \
#     -gt_path /mnt/disk1/hyein99/datasets/MSSPACEV1B/public_query_gt100.bin \
#     -data_type int8 \
#     -dist_func Euclidian > spacev_1b.log 2>&1