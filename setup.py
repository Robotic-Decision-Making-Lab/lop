from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='lop',
      version='0.0.1',
      description='Learn Objective functions from Preferences (lop)',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/ianran/rdml_graph',
      author='Ian Rankin',
      author_email='rankini@oregonstate.edu',
      license='MIT',
      #packages=['rdml_graph', 'rdml_graph.core'],
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      install_requires=['numpy','matplotlib', 'scipy', 'tqdm>=3.0.0', 'oyaml>=1.0.0', 'pytest'],
      #extras_require={'Saving graphs': ["pickle"]},
      python_requires='>=2.7',
      zip_safe=False)
