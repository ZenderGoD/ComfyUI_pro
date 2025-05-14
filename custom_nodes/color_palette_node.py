import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from collections import Counter
import colorsys
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import math
import cv2
from typing import List, Tuple, Dict, Any

class ColorAnalysis:
    """Advanced color analysis utilities"""
    
    @staticmethod
    def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    
    @staticmethod
    def rgb_to_lab(rgb: Tuple[int, int, int]) -> LabColor:
        rgb_color = sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        return convert_color(rgb_color, LabColor)
    
    @staticmethod
    def get_color_temperature(rgb: Tuple[int, int, int]) -> str:
        h, s, v = ColorAnalysis.rgb_to_hsv(rgb)
        h_deg = h * 360
        
        if v < 0.2:
            return "Dark"
        elif v > 0.8:
            return "Light"
        elif s < 0.2:
            return "Neutral"
        elif 30 <= h_deg <= 90:
            return "Warm"
        elif 210 <= h_deg <= 270:
            return "Cool"
        elif 90 < h_deg < 210:
            return "Cool"
        else:
            return "Warm"
    
    @staticmethod
    def get_color_harmony(colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """Calculate color harmony metrics"""
        harmonies = {
            "complementary": 0.0,
            "analogous": 0.0,
            "triadic": 0.0,
            "split_complementary": 0.0,
            "tetradic": 0.0
        }
        
        if len(colors) < 2:
            return harmonies
            
        hsv_colors = [ColorAnalysis.rgb_to_hsv(c) for c in colors]
        hues = [h * 360 for h, _, _ in hsv_colors]
        
        # Complementary (180° apart)
        for i, h1 in enumerate(hues):
            for h2 in hues[i+1:]:
                diff = abs(h1 - h2)
                if 175 <= diff <= 185:
                    harmonies["complementary"] += 1
                    
        # Analogous (30° apart)
        for i, h1 in enumerate(hues):
            for h2 in hues[i+1:]:
                diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
                if diff <= 30:
                    harmonies["analogous"] += 1
                    
        # Normalize scores
        total = len(colors) * (len(colors) - 1) / 2
        if total > 0:
            for k in harmonies:
                harmonies[k] /= total
                
        return harmonies
    
    @staticmethod
    def get_contrast_ratio(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate WCAG contrast ratio between two colors"""
        def get_luminance(rgb):
            r, g, b = [c/255 for c in rgb]
            r = r/12.92 if r <= 0.03928 else ((r + 0.055)/1.055) ** 2.4
            g = g/12.92 if g <= 0.03928 else ((g + 0.055)/1.055) ** 2.4
            b = b/12.92 if b <= 0.03928 else ((b + 0.055)/1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
            
        l1 = get_luminance(color1)
        l2 = get_luminance(color2)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)
    
    @staticmethod
    def get_accessibility_scores(colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Calculate WCAG accessibility scores for color combinations"""
        scores = {
            "AA": {"pass": 0, "total": 0},
            "AAA": {"pass": 0, "total": 0}
        }
        
        for i, c1 in enumerate(colors):
            for c2 in colors[i+1:]:
                ratio = ColorAnalysis.get_contrast_ratio(c1, c2)
                scores["AA"]["total"] += 1
                scores["AAA"]["total"] += 1
                
                if ratio >= 4.5:  # WCAG AA standard
                    scores["AA"]["pass"] += 1
                if ratio >= 7.0:  # WCAG AAA standard
                    scores["AAA"]["pass"] += 1
                    
        return scores

class ColorPaletteExtractor:
    """Advanced color palette extraction with multiple methods and analysis"""
    
    def __init__(self):
        self.methods = {
            "kmeans": self._extract_kmeans,
            "median_cut": self._extract_median_cut,
            "color_quantization": self._extract_color_quantization
        }
    
    def _extract_kmeans(self, image: Image.Image, num_colors: int) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Extract colors using K-means clustering"""
        arr = np.array(image).reshape(-1, 3)
        model = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
        labels = model.fit_predict(arr)
        counts = Counter(labels)
        palette = model.cluster_centers_.astype(int)
        total = sum(counts.values())
        percentages = [round((counts[i] / total) * 100, 2) for i in range(len(palette))]
        return palette.tolist(), percentages
    
    def _extract_median_cut(self, image: Image.Image, num_colors: int) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Extract colors using median cut algorithm"""
        arr = np.array(image).reshape(-1, 3)
        pixels = arr.tolist()
        
        def get_channel_range(pixels, channel):
            return max(p[channel] for p in pixels) - min(p[channel] for p in pixels)
        
        def split_bucket(pixels, depth=0):
            if len(pixels) == 0 or depth >= math.log2(num_colors):
                return [pixels]
                
            channel = max(range(3), key=lambda c: get_channel_range(pixels, c))
            pixels.sort(key=lambda p: p[channel])
            median_idx = len(pixels) // 2
            return split_bucket(pixels[:median_idx], depth+1) + split_bucket(pixels[median_idx:], depth+1)
        
        buckets = split_bucket(pixels)
        palette = []
        percentages = []
        total_pixels = len(arr)
        
        for bucket in buckets:
            if not bucket:
                continue
            avg_color = tuple(map(int, np.mean(bucket, axis=0)))
            palette.append(avg_color)
            percentages.append(round((len(bucket) / total_pixels) * 100, 2))
            
        return palette, percentages
    
    def _extract_color_quantization(self, image: Image.Image, num_colors: int) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Extract colors using color quantization"""
        img_np = np.array(image)
        pixels = img_np.reshape(-1, 3)
        
        # Convert to float32 for kmeans
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        counts = Counter(labels.flatten())
        total = sum(counts.values())
        percentages = [round((counts[i] / total) * 100, 2) for i in range(len(centers))]
        
        return centers.tolist(), percentages
    
    def get_palette(self, image: Image.Image, num_colors: int, method: str = "kmeans") -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Get color palette using specified method"""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(self.methods.keys())}")
        return self.methods[method](image, num_colors)
    
    def analyze_colors(self, palette: List[Tuple[int, int, int]], percentages: List[float]) -> Dict[str, Any]:
        """Perform comprehensive color analysis"""
        analysis = {
            "temperature": {},
            "harmony": ColorAnalysis.get_color_harmony(palette),
            "accessibility": ColorAnalysis.get_accessibility_scores(palette),
            "color_details": []
        }
        
        # Analyze each color
        for color, percentage in zip(palette, percentages):
            h, s, v = ColorAnalysis.rgb_to_hsv(color)
            lab = ColorAnalysis.rgb_to_lab(color)
            
            color_info = {
                "rgb": color,
                "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                "percentage": percentage,
                "temperature": ColorAnalysis.get_color_temperature(color),
                "hsv": (h, s, v),
                "lab": (lab.lab_l, lab.lab_a, lab.lab_b)
            }
            
            # Update temperature distribution
            temp = color_info["temperature"]
            analysis["temperature"][temp] = analysis["temperature"].get(temp, 0) + percentage
            
            analysis["color_details"].append(color_info)
            
        return analysis
    
    def generate_variants(self, palette: List[Tuple[int, int, int]], percentages: List[float]) -> Dict[str, List[Tuple[Tuple[int, int, int], float]]]:
        """Generate color variants and groupings"""
        variants = {
            "Temperature": {"Warm": [], "Cool": [], "Neutral": [], "Light": [], "Dark": []},
            "Harmony": {"Complementary": [], "Analogous": [], "Triadic": []},
            "Accessibility": {"High Contrast": [], "Medium Contrast": [], "Low Contrast": []}
        }
        
        # Group by temperature
        for color, percentage in zip(palette, percentages):
            temp = ColorAnalysis.get_color_temperature(color)
            variants["Temperature"][temp].append((color, percentage))
            
        # Group by harmony
        hsv_colors = [ColorAnalysis.rgb_to_hsv(c) for c in palette]
        hues = [h * 360 for h, _, _ in hsv_colors]
        
        for i, (color1, p1) in enumerate(zip(palette, percentages)):
            for j, (color2, p2) in enumerate(zip(palette[i+1:], percentages[i+1:])):
                h1, h2 = hues[i], hues[j+i+1]
                diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
                
                if 175 <= diff <= 185:
                    variants["Harmony"]["Complementary"].extend([(color1, p1), (color2, p2)])
                elif diff <= 30:
                    variants["Harmony"]["Analogous"].extend([(color1, p1), (color2, p2)])
                elif 115 <= diff <= 125:
                    variants["Harmony"]["Triadic"].extend([(color1, p1), (color2, p2)])
                    
        # Group by accessibility
        for i, (color1, p1) in enumerate(zip(palette, percentages)):
            for color2, p2 in zip(palette[i+1:], percentages[i+1:]):
                ratio = ColorAnalysis.get_contrast_ratio(color1, color2)
                if ratio >= 7.0:
                    variants["Accessibility"]["High Contrast"].extend([(color1, p1), (color2, p2)])
                elif ratio >= 4.5:
                    variants["Accessibility"]["Medium Contrast"].extend([(color1, p1), (color2, p2)])
                else:
                    variants["Accessibility"]["Low Contrast"].extend([(color1, p1), (color2, p2)])
                    
        return variants
    
    def render_palette_image(self, variants: Dict[str, Dict[str, List[Tuple[Tuple[int, int, int], float]]]], 
                           analysis: Dict[str, Any]) -> Image.Image:
        """Render an advanced palette visualization"""
        # Create a larger image to accommodate all information
        width = 800
        section_height = 200
        total_height = section_height * (len(variants) + 2)  # +2 for analysis and harmony sections
        img = Image.new("RGB", (width, total_height), "white")
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        y_offset = 0
        
        # Draw analysis section
        draw.text((10, y_offset), "Color Analysis", font=title_font, fill="black")
        y_offset += 30
        
        # Temperature distribution
        temp_text = "Temperature Distribution: " + ", ".join(
            f"{temp}: {pct:.1f}%" for temp, pct in analysis["temperature"].items()
        )
        draw.text((20, y_offset), temp_text, font=font, fill="black")
        y_offset += 30
        
        # Harmony scores
        harmony_text = "Harmony Scores: " + ", ".join(
            f"{harmony}: {score:.2f}" for harmony, score in analysis["harmony"].items()
        )
        draw.text((20, y_offset), harmony_text, font=font, fill="black")
        y_offset += 30
        
        # Accessibility scores
        acc_text = "WCAG Compliance: " + ", ".join(
            f"{level} AA: {scores['pass']}/{scores['total']}" 
            for level, scores in analysis["accessibility"].items()
        )
        draw.text((20, y_offset), acc_text, font=font, fill="black")
        y_offset += 50
        
        # Draw each variant section
        for category, groups in variants.items():
            draw.text((10, y_offset), f"{category} Variants", font=title_font, fill="black")
            y_offset += 30
            
            for group_name, colors in groups.items():
                if not colors:
                    continue
                    
                draw.text((20, y_offset), f"{group_name}:", font=font, fill="black")
                x = 150
                
                for color, percentage in colors:
                    # Draw color box
                    draw.rectangle([x, y_offset, x+30, y_offset+30], fill=color)
                    # Draw percentage
                    draw.text((x+35, y_offset), f"{percentage:.1f}%", font=font, fill="black")
                    x += 80
                    
                y_offset += 40
                
            y_offset += 20
            
        return img
    
    def get_text_summary(self, variants: Dict[str, Dict[str, List[Tuple[Tuple[int, int, int], float]]]], 
                        analysis: Dict[str, Any]) -> str:
        """Generate a detailed text summary of the palette analysis"""
        lines = []
        
        # Analysis summary
        lines.append("=== Color Analysis ===")
        lines.append(f"Temperature Distribution: {', '.join(f'{temp}: {pct:.1f}%' for temp, pct in analysis['temperature'].items())}")
        lines.append(f"Harmony Scores: {', '.join(f'{harmony}: {score:.2f}' for harmony, score in analysis['harmony'].items())}")
        lines.append(f"WCAG Compliance: {', '.join(f'{level} AA: {scores['pass']}/{scores['total']}' for level, scores in analysis['accessibility'].items())}")
        lines.append("")
        
        # Color details
        lines.append("=== Color Details ===")
        for color_info in analysis["color_details"]:
            lines.append(
                f"Color: {color_info['hex']} ({color_info['temperature']}) - {color_info['percentage']}%"
            )
        lines.append("")
        
        # Variants summary
        lines.append("=== Color Variants ===")
        for category, groups in variants.items():
            lines.append(f"\n{category}:")
            for group_name, colors in groups.items():
                if colors:
                    color_str = ", ".join(f"{'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'} ({p:.1f}%)" for c, p in colors)
                    lines.append(f"  {group_name}: {color_str}")
                    
        return "\n".join(lines)

class ColorPaletteNode:
    """ComfyUI node for advanced color palette extraction"""
    
    def __init__(self):
        self.extractor = ColorPaletteExtractor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 6,
                    "min": 3,
                    "max": 12,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of colors to extract"
                }),
                "method": (["kmeans", "median_cut", "color_quantization"], {
                    "default": "kmeans",
                    "tooltip": "Color extraction method"
                }),
            },
            "optional": {
                "show_analysis": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Show detailed color analysis"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("palette_preview", "analysis_summary",)
    FUNCTION = "extract"
    CATEGORY = "image/color"
    OUTPUT_NODE = False
    
    def extract(self, image: torch.Tensor, num_colors: int, method: str, show_analysis: str = "enable") -> Tuple[torch.Tensor, str]:
        """Extract and analyze color palette from input image"""
        # Convert ComfyUI tensor image to PIL
        image = (image[0] * 255).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        # Extract palette
        palette, percentages = self.extractor.get_palette(pil_image, num_colors, method)
        
        # Analyze colors
        analysis = self.extractor.analyze_colors(palette, percentages)
        
        # Generate variants
        variants = self.extractor.generate_variants(palette, percentages)
        
        # Generate output
        palette_img = self.extractor.render_palette_image(variants, analysis)
        text_summary = self.extractor.get_text_summary(variants, analysis)
        
        # Convert back to tensor
        palette_img = palette_img.resize((512, 512))
        out_tensor = np.array(palette_img).astype(np.float32) / 255.0
        out_tensor = np.expand_dims(out_tensor, axis=0)
        out_tensor = torch.from_numpy(out_tensor)
        
        return (out_tensor, text_summary)
    
    @classmethod
    def IS_CHANGED(cls, image, num_colors, method, show_analysis):
        """Force re-execution when inputs change"""
        return ""

# Node registration - must be at the end of the file
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

NODE_CLASS_MAPPINGS = {
    "ColorPaletteadv": ColorPaletteNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPaletteadv": "🎨 Color Palette Extractor"
}

# Make sure these are the last lines in the file
