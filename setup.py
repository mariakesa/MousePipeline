from setuptools import setup, find_packages

setup(
    name='vit_pipeline',
    version='0.1.0',
    packages=find_packages(include=['vit_pipeline', 'vit_pipeline.*']),
    author='Maria Kesa',
    author_email='mariarosekesa@gmail.com',
    description='STA computation code for VIT embeddings',
    url='https://github.com/MousePipeline',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
