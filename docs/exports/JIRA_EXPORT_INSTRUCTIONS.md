
# How to Export UML Diagrams for Jira

## Option 1: Browser Screenshot (Recommended)
1. Open each HTML file in your browser:
   - docs/exports/system_overview.html
   - docs/exports/sequence_diagram.html
   - docs/exports/camera_service.html
   - docs/exports/deployment.html

2. Take screenshots of each diagram
3. Upload to Jira as images

## Option 2: Browser Developer Tools
1. Open HTML file in browser
2. Right-click on diagram → "Inspect Element"
3. Find the SVG element
4. Right-click → "Save image as..."

## Option 3: Using Puppeteer (Automated)
Run: npm install puppeteer
Then use the provided script to auto-export PNG files.

## Option 4: Mermaid Live Editor
1. Go to https://mermaid.live/
2. Copy diagram code from docs/architecture_uml.md
3. Export as PNG/SVG

## Files Created:
- system_overview.html - High-level architecture
- sequence_diagram.html - Service interactions
- camera_service.html - Camera service details
- deployment.html - Infrastructure setup

## For Jira:
- Upload as images to your Jira dashboard
- Use in documentation pages
- Add to project wikis
- Include in sprint planning
