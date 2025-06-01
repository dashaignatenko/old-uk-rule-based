from setuptools import setup, find_packages
import os

setup(
    name="old_uk_parser",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pyconll>=3.2.0',
        'fuzzywuzzy>=0.18.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'nltk>=3.8.0',
        'scikit-learn>=1.2.0',
        'scipy>=1.10.0',
        'sklearn-crfsuite>=0.3.6'
    ],
    include_package_data=True, 
    package_data={"old_uk_parser": ["*.json"], "old_uk_parser": ["*.py"]},
    author="Daria Ignatenko",
    author_email="ignadash@gmail.com",
    description="Rule-based morphological parser for Old Ukrainian",
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else "Rule-based morphological parser for Old Ukrainian",
    long_description_content_type='text/markdown',
    url="https://github.com/dashaignatenko/old-uk-rule-based",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
