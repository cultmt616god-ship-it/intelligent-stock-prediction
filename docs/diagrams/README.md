# Diagram Gallery

This directory contains a collection of Mermaid diagrams documenting the Stock Market Prediction System architecture and design.

## Gallery

The `gallery.html` file provides an interactive web-based gallery showcasing all diagrams with a neutral theme. To view the gallery:

1. Open `gallery.html` in a web browser
2. All diagrams are rendered directly in the browser using Mermaid.js

## Diagrams Included

1. **Level 0 Data Flow Diagram** (`dfd_level0.md`) - High-level system data flows
2. **Portfolio Management DFD (Level 1)** (`dfd_portfolio_level1.md`) - Detailed portfolio management processes
3. **Entity Relationship Diagram** (`er_diagram.md`) - Database schema and relationships
4. **Use Case Diagram** (`usecase_diagram.md`) - Functional requirements by user role
5. **Project Gantt Chart** (`project_gantt.md`) - Project timeline and phases

## Exporting Diagrams

To export diagrams as PNG and SVG images with a 16:9 aspect ratio:

1. Install required dependencies:
   ```bash
   pip install requests
   ```

2. (Optional) Install Mermaid CLI for higher quality exports:
   ```bash
   npm install -g @mermaid-js/mermaid-cli
   ```

3. Run the export script:
   ```bash
   python export_diagrams.py
   ```

4. Find the exported images in the `exported/` directory

The export script will create both PNG and SVG versions of each diagram with a 16:9 aspect ratio using a neutral theme.