from setuptools import setup, find_packages

setup(
    name='traffictelligence',
    version='0.1.0',
    description='Advanced Traffic Volume Estimation using Machine Learning',
    author='Keerthi Reddy S',
    author_email='youremail@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'joblib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.7',
)
