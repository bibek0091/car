import matplotlib.pyplot as plt
from svgpathtools import svg2paths
import numpy as np

def test_map(svg_path):
    print(f"Loading {svg_path}...")
    paths, attributes = svg2paths(svg_path)
    
    print(f"Found {len(paths)} paths in SVG.")
    
    plt.figure(figsize=(10, 8))
    for i, path in enumerate(paths):
        points = []
        num_points = int(path.length() / 10.0) + 1
        for j in range(num_points):
            t = j / max(1, (num_points - 1))
            pt = path.point(t)
            points.append([pt.real, pt.imag])
        
        if points:
            points = np.array(points)
            plt.plot(points[:, 0], -points[:, 1], label=f'Path {i}')  # Invert Y for correct visualization

    plt.title("Track2023 SVG Parsed")
    plt.axis('equal')
    plt.savefig('map_preview.png')
    print("Saved preview to map_preview.png")

if __name__ == "__main__":
    test_map('Track.svg')
