git clone https://github.com/ariel415el/STRAPS.git
cd STRAPS
pip install gdown

gdown --folder https://drive.google.com/drive/folders/1CLOqQBrTos7vhohjFcU2OFkNYmyvQf6t # Downlaod data
mv training\ data/* data/
rm -rf training\ data/

gdown --folder https://drive.google.com/drive/folders/1phJix1Fp-AbJgoLImb19eXCWEK7ZnAp_ # Downlaod additional files
mv additional\ files/* additional
rm -rf additional\ files

gdown --fuzzy 1xYIXOuuuTTuwbcvAByitNdYv1clDmrmZ        # download smpl model
apt install unzip
mkdir additional/SMPL/
unzip SMPL.zip -d additional/SMPL/
rm -rf SMPL.zip

apt-get update && sudo apt-get install -y libgl1
pip install -r requirements.txt
#pip install chumpy
pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17 # This solve some import error

## for prediction
#python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#pip install git+https://github.com/facebookresearch/segment-anything.git

git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer

# Aply fixes for cuda
OUTPUT_FILE="fix_torch_check.diff"
# Create the file and write the content
cat << 'EOF' > $OUTPUT_FILE
diff --git a/neural_renderer/cuda/create_texture_image_cuda.cpp b/neural_renderer/cuda/create_texture_image_cuda.cpp
index c2c449d..85d287f 100644
--- a/neural_renderer/cuda/create_texture_image_cuda.cpp
+++ b/neural_renderer/cuda/create_texture_image_cuda.cpp
@@ -10,8 +10,8 @@ at::Tensor create_texture_image_cuda(

 // C++ interface

-#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
-#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
+#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
+#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
 #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


diff --git a/neural_renderer/cuda/load_textures_cuda.cpp b/neural_renderer/cuda/load_textures_cuda.cpp
index 8cb99bc..5e26a54 100644
--- a/neural_renderer/cuda/load_textures_cuda.cpp
+++ b/neural_renderer/cuda/load_textures_cuda.cpp
@@ -12,8 +12,8 @@ at::Tensor load_textures_cuda(

 // C++ interface

-#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
-#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
+#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
+#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
 #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


diff --git a/neural_renderer/cuda/rasterize_cuda.cpp b/neural_renderer/cuda/rasterize_cuda.cpp
index ce7a3e5..7ecebd3 100644
--- a/neural_renderer/cuda/rasterize_cuda.cpp
+++ b/neural_renderer/cuda/rasterize_cuda.cpp
@@ -63,8 +63,8 @@ at::Tensor backward_depth_map_cuda(

 // C++ interface

-#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
-#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
+#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
+#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
 #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

 std::vector<at::Tensor> forward_face_index_map(
EOF

git apply fix_torch_check.diff
pip install .



