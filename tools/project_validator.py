"""
BIA601 Project Validator
This tool validates the project against course requirements
"""

import os
import ast
import re
from typing import List, Dict, Set

class ProjectValidator:
    def __init__(self):
        self.allowed_libraries = {
            'numpy', 'pandas', 'sklearn', 'matplotlib', 
            'seaborn', 'streamlit', 'plotly', 'pytest'
        }
        
        self.forbidden_libraries = {
            'tensorflow', 'torch', 'keras', 'theano',
            'pygad', 'deap', 'neat-python'
        }
    
    def validate_file(self, file_path: str) -> Dict:
        """Validate a single Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = {
            'arabic_text': self._check_arabic(content),
            'imports': self._check_imports(content),
            'forbidden_patterns': self._check_forbidden_patterns(content)
        }
        return results
    
    def _check_arabic(self, content: str) -> List[str]:
        """Check for Arabic text in content"""
        arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
        matches = arabic_pattern.findall(content)
        return matches
    
    def _check_imports(self, content: str) -> Dict:
        """Analyze imports in the file"""
        try:
            tree = ast.parse(content)
            imports = {
                'allowed': set(),
                'forbidden': set(),
                'unknown': set()
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        self._categorize_import(name.name, imports)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._categorize_import(node.module, imports)
            
            return imports
        except:
            return {'error': 'Could not parse file'}
    
    def _categorize_import(self, name: str, imports: Dict[str, Set]):
        """Categorize an import as allowed, forbidden, or unknown"""
        base_module = name.split('.')[0]
        if base_module in self.allowed_libraries:
            imports['allowed'].add(base_module)
        elif base_module in self.forbidden_libraries:
            imports['forbidden'].add(base_module)
        else:
            imports['unknown'].add(base_module)
    
    def _check_forbidden_patterns(self, content: str) -> List[str]:
        """Check for forbidden code patterns"""
        forbidden_patterns = [
            (r'eval\s*\(', 'Use of eval() function'),
            (r'exec\s*\(', 'Use of exec() function'),
            (r'\.fit\s*\([^)]*epochs', 'Neural network training pattern detected'),
            (r'nn\.', 'Neural network module usage detected'),
            (r'layer', 'Possible neural network layer definition')
        ]
        
        matches = []
        for pattern, msg in forbidden_patterns:
            if re.search(pattern, content):
                matches.append(msg)
        return matches

def validate_project(project_path: str) -> Dict:
    """
    Validate entire project directory
    Returns a detailed report of compliance
    """
    validator = ProjectValidator()
    report = {
        'files_checked': 0,
        'violations': [],
        'compliance_score': 100,
        'recommendations': []
    }
    
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                results = validator.validate_file(file_path)
                
                # Process results
                if results['arabic_text']:
                    report['violations'].append({
                        'file': file_path,
                        'type': 'Arabic text found',
                        'details': results['arabic_text']
                    })
                
                if 'error' not in results['imports']:
                    if results['imports']['forbidden']:
                        report['violations'].append({
                            'file': file_path,
                            'type': 'Forbidden imports',
                            'details': list(results['imports']['forbidden'])
                        })
                
                if results['forbidden_patterns']:
                    report['violations'].append({
                        'file': file_path,
                        'type': 'Forbidden patterns',
                        'details': results['forbidden_patterns']
                    })
                
                report['files_checked'] += 1
    
    # Calculate compliance score
    if report['violations']:
        report['compliance_score'] -= len(report['violations']) * 5
        report['compliance_score'] = max(0, report['compliance_score'])
    
    # Add recommendations
    if report['violations']:
        report['recommendations'].append(
            "Remove Arabic text and comments from code files"
        )
        if any(v['type'] == 'Forbidden imports' for v in report['violations']):
            report['recommendations'].append(
                "Replace forbidden libraries with approved alternatives"
            )
    
    return report

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report = validate_project(project_path)
    
    print("\n=== BIA601 Project Validation Report ===")
    print(f"\nFiles Checked: {report['files_checked']}")
    print(f"Compliance Score: {report['compliance_score']}%")
    
    if report['violations']:
        print("\nViolations Found:")
        for v in report['violations']:
            print(f"\n- File: {os.path.basename(v['file'])}")
            print(f"  Type: {v['type']}")
            print(f"  Details: {v['details']}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")