import json
from pathlib import Path


# Define notebook categories. Note the order is retained in the website. 
MODELS_NOTEBOOKS = [
    'ipw.ipynb',
    'standardization.ipynb',
    'doubly_robust.ipynb',
    'TMLE.ipynb',
    'rlearner.ipynb',
    'xlearner.ipynb',
    'causal_survival_analysis.ipynb',
    'matching.ipynb',
    'matching_with_custom_backends.ipynb',
    'lalonde_matching.ipynb',
    'hemm_demo.ipynb',
    'positivity.ipynb',
    'evaluation_plots_overview.ipynb',
]

USE_CASES_NOTEBOOKS = [
    'causal_inference_vs_descriptive_statistics.ipynb',
    'Bank-Marketing.ipynb',
    'Dehejia_Wahba_replication.ipynb',
    'nhefs.ipynb',
    'MANAGE agricultural data.ipynb',
    'fast_food_employment_card_krueger.ipynb',
    'lalonde.ipynb',
    'causal_simulator.ipynb',
]


def generate_examples_gallery():
    """
    Generate a dynamic gallery of Jupyter notebook examples.
    Scans the examples directory and creates an index.md file with links to all notebooks.
    Organizes notebooks into three sections: Models, Use Cases, and Miscellaneous.
    Also creates symlinks to notebooks in the docs/source/examples directory for nbsphinx.
    """
    
    # Get the directory where conf.py is located
    conf_dir = Path(__file__).resolve().parent
    
    # Paths relative to conf.py location (docs/source/)
    examples_source_dir = (conf_dir / ".." / ".." / "examples").resolve()  # Root examples directory
    examples_doc_dir = (conf_dir / "examples").resolve()  # Sphinx examples directory
    index_file = examples_doc_dir / "index.md"
    
    # Create examples directory if it doesn't exist
    examples_doc_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all notebooks
    notebooks = examples_source_dir.glob("*.ipynb")
    
    if not notebooks:
        print(f"Warning: No notebooks found in {examples_source_dir}")
        return

    # Order the notebooks according to the pre-specified lists, but don't forget if anything is missing:
    order = {name: i for i, name in enumerate(USE_CASES_NOTEBOOKS + MODELS_NOTEBOOKS)}
    notebooks = sorted(notebooks, key=lambda p: order.get(p.name, float('inf')))
    
    # Extract metadata from each notebook and create symlinks
    # Organize into categories
    models_info = []
    use_cases_info = []
    misc_info = []

    for notebook_path in notebooks:
        try:
            nb_data = json.loads(notebook_path.read_text())
            
            # Extract title from first markdown cell
            title = None
            description = None
            for cell in nb_data.get('cells', []):
                if cell.get('cell_type') == 'markdown':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        source = ''.join(source)
                    
                    # Look for title (lines starting with #)
                    lines = source.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('#'):
                            # Remove markdown heading markers
                            title = line.lstrip('#').strip()
                            break
                    
                    # Get description (first non-heading line)
                    if title:
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                description = line
                                break
                    break
            
            # Fallback to filename if no title found
            if not title:
                title = notebook_path.stem.replace('_', ' ').title()
            
            # Create symlink in docs/source/examples/ pointing to the actual notebook
            symlink_path = examples_doc_dir / notebook_path.name
            # Compute relative path from symlink location to actual notebook
            target_path = Path("..") / ".." / ".." / "examples" / notebook_path.name
            
            # Remove existing symlink/file if it exists
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            
            # Create symlink
            try:
                symlink_path.symlink_to(target_path)
            except OSError:
                # If symlink fails (e.g., on Windows), copy the file instead
                import shutil
                shutil.copy2(notebook_path, symlink_path)
            
            info = {
                'title': title,
                'description': description or '',
                'filename': notebook_path.name,
                'path': notebook_path.stem
            }
            
            # Categorize the notebook
            if notebook_path.name in MODELS_NOTEBOOKS:
                models_info.append(info)
            elif notebook_path.name in USE_CASES_NOTEBOOKS:
                use_cases_info.append(info)
            else:
                misc_info.append(info)
            
        except Exception as e:
            print(f"Warning: Could not process {notebook_path}: {e}")
            continue
    
    # Combine all notebooks for toctree
    all_notebook_info = models_info + use_cases_info + misc_info
    
    # Generate the index.md content
    content = [
        "# Examples Gallery",
        "",
        "This gallery showcases various examples demonstrating the capabilities of CausalLib.",
        "Each example is a Jupyter notebook that you can view, download, and run locally.",
        "",
        "```{eval-rst}",
        ".. toctree::",
        "   :maxdepth: 1",
        "   :hidden:",
        "",
    ]
    
    # Add toctree entries for all notebooks
    for info in all_notebook_info:
        content.append(f"   {info['path']}")
    
    content.extend([
        "```",
        "",
        "---",
        "",
    ])
    
    # Add sections
    sections = [
        ("Models and Features", models_info),
        ("Real-World Use Cases", use_cases_info),
    ]
    
    if misc_info:
        sections.append(("Miscellaneous", misc_info))
    
    for section_title, section_info in sections:
        if not section_info:
            continue
            
        content.append(f"## {section_title}")
        content.append("")
        content.append("::::{grid} 1 1 2 2")
        content.append(":gutter: 3")
        content.append("")
        
        for info in section_info:
            content.append(f":::{{grid-item-card}} {info['title']}")
            content.append(f":link: {info['path']}")
            content.append(":link-type: doc")
            content.append("")
            if info['description']:
                # Truncate long descriptions
                desc = info['description']
                if len(desc) > 150:
                    desc = desc[:147] + "..."
                content.append(desc)
            content.append(":::")
            content.append("")
        
        content.append("::::")
        content.append("")
    
    # Write the index file
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text('\n'.join(content), encoding='utf-8')
    
    print(f"Generated examples gallery with {len(all_notebook_info)} notebooks")
    print(f"  - Models and Features: {len(models_info)}")
    print(f"  - Real-World Use Cases: {len(use_cases_info)}")
    print(f"  - Miscellaneous: {len(misc_info)}")

# Assisted by Bob