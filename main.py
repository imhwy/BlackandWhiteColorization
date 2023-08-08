import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_path", type=str,
                    default="a.jpg")
parser.add_argument("--use_gpu", action="store_true",
                    help="whether to use GPU")
parser.add_argument(
    "-o",
    "--save_prefix",
    type=str,
    default="saved",
    help="will save into this file with {base.png, improvement.png} suffixes",
)
opt = parser.parse_args()

# load colorizers
colorizer_base = base(pretrained=True).eval()
colorizer_improvement = improvement(pretrained=True).eval()
if opt.use_gpu:
    colorizer_base.cuda()
    colorizer_improvement.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
if opt.use_gpu:
    tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(
    tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1)
)
out_img_base = postprocess_tens(tens_l_orig, colorizer_base(tens_l_rs).cpu())
out_img_improvement = postprocess_tens(
    tens_l_orig, colorizer_improvement(tens_l_rs).cpu()
)

plt.imsave("%s_base.png" % opt.save_prefix, out_img_base)
plt.imsave("%s_improvement.png" % opt.save_prefix, out_img_improvement)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(img_bw)
plt.title("Input")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(out_img_base)
plt.title("Output (base)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(out_img_improvement)
plt.title("Output (improvement)")
plt.axis("off")
plt.show()
