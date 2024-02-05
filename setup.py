import setuptools
import os
path = os.path.join(os.path.dirname(__file__), "python")
if __name__ == '__main__':
    setuptools.setup(
        name='JDiffusion',
        version='1.0',
        description='',
        author='',
        author_email='',
        packages=setuptools.find_packages(path),
        package_dir={'': "python"},
        install_requires=[
            # 'cupy',
        ]
    )