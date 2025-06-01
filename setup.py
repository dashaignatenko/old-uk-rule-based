from setuptools import setup, find_packages

setup(
    name="OldUk_morph",
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
    package_data={
        'old_uk_parser': ['*.json']
    },
    author="Daria Ignatenko",
    author_email="ignadash@gmail.com",
    description="Rule-based morphological parser for Old Ukrainian",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/dashaignatenko/old-uk-rule-based",
)