import json
from pathlib import Path
from typing import Dict, Any

class AgentReport:
    """Generates a markdown report for agent verification."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_report(self, 
                       metrics: Dict[str, Any], 
                       composite_image_path: Path, 
                       filename: str = "verification_report.md") -> Path:
        """
        Generate a markdown report.
        
        Args:
            metrics: Dictionary of metrics from DataValidator.
            composite_image_path: Path to the visual composite.
            filename: Output filename.
            
        Returns:
            Path to the saved report.
        """
        report_path = self.output_dir / filename
        
        # Relative path for the image in the markdown
        rel_image_path = composite_image_path.relative_to(self.output_dir)
        
        content = f"""# Verification Report

## Visual Inspection
![Composite View]({rel_image_path})

## Data Metrics
```json
{json.dumps(metrics, indent=4)}
```

## Agent Assessment Instructions
1.  **Visual Check**: Look at the "Composite View".
    *   **Sky**: Is the sky region flat (bad) or infinite/far (good)?
    *   **Edges**: Are object boundaries sharp in the Depth Map?
    *   **Confidence**: Does the Confidence Map align with difficult areas (edges, transparent objects)?
2.  **Data Check**: Review the JSON metrics.
    *   `high_confidence_ratio`: Should be high (> 0.8) for good quality.
    *   `edge_sharpness_mean`: Higher is generally better.
3.  **Conclusion**: Based on the above, determine if the depth estimation is satisfactory.
"""
        with open(report_path, "w") as f:
            f.write(content)
            
        return report_path
