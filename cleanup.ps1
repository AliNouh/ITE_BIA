# Clean temporary and generated files
Get-ChildItem -Path . -Recurse -Include "__pycache__" | Remove-Item -Force -Recurse
Get-ChildItem -Path . -Recurse -Include "*.pyc","*.pyo","*.pyd","*.so" | Remove-Item -Force
Remove-Item -Path "ite_bia.egg-info" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path ".pytest_cache" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "build" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "dist" -Force -Recurse -ErrorAction SilentlyContinue

# Clean and unify documentation
$docsPath = "docs"
if (Test-Path -Path "$docsPath\report_arabic_backup.md") {
    Remove-Item -Path "$docsPath\report_arabic_backup.md" -Force
}

# Remove unnecessary files
Remove-Item -Path "demo_simple.py" -Force -ErrorAction SilentlyContinue