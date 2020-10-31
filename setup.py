from setuptools import setup, find_packages
setup(
    name='SALaT',
    version='1.0.0',
    author='You Zhou',
    author_email='you.zhou@kit.edu',
    description="Shift Attention Latent Transformation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'shapely'
    ]
)
