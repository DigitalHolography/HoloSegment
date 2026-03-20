import imageio
from holosegment.utils.image_utils import normalize_to_uint8, save_numpy_as_avi, save_labeled_branches
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

class DebugRenderer:
    def render(self, key, cache, path):
        raise NotImplementedError
    
class ImageRenderer(DebugRenderer):
    def render(self, key, cache, path):
        imageio.imwrite(path, normalize_to_uint8(cache.get(key)))

class ImageColorbarRenderer(DebugRenderer):

    def save_image_with_colorbar(self, path, image_data):
        norm = Normalize(vmin=np.min(image_data), vmax=np.max(image_data))
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(image_data, cmap='viridis', norm=norm)
        fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
        ax.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
        plt.close(fig)

    def render(self, key, cache, path):
        save_image_with_colorbar(path, cache.get(key))

class SignalRenderer(DebugRenderer):
    def render(self, key, cache, path):
        plt.figure()
        plt.plot(cache.get(key))
        plt.title(key)
        plt.savefig(path)
        plt.close()

class VideoRenderer(DebugRenderer):
    def render(self, key, cache, path):
        save_numpy_as_avi(cache.get(key), path.with_suffix(".avi"))

class OpticDiscRenderer(DebugRenderer):
    def render(self, key, cache, path):
        image = cache.get("M0_ff_image")
        center = cache.get("optic_disc_center")
        axes = cache.get("optic_disc_axes")

        x_center, y_center = center
        diameter_x, diameter_y = axes

        a = diameter_x / 2
        b = diameter_y / 2

        angle = np.linspace(0, 2*np.pi, 360)

        x = x_center + a*np.cos(angle)
        y = y_center + b*np.sin(angle)

        plt.figure(figsize=(6,6))
        plt.imshow(image, cmap="gray")

        plt.plot(x, y, "r")

        plt.gca().add_patch(
            plt.Rectangle(
                (x_center-a, y_center-b),
                diameter_x,
                diameter_y,
                fill=False,
                edgecolor="lime"
            )
        )

        plt.savefig(path)
        plt.close()

class LabeledMaskRenderer(DebugRenderer):
    def render(self, key, cache, path):
        save_labeled_branches(cache.get(key), path)
        plt.close()