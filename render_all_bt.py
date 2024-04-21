import pybts
import os
from bt import BTBuilder
from pybts.display import render_node

"""
生成scripts中所有行为树的图到scripts/images中
"""

SCRIPTS_PATH = 'scripts'
IMAGES_PATH = 'scripts/images'

if __name__ == '__main__':
    builder = BTBuilder()
    for filename in os.listdir(SCRIPTS_PATH):
        if not filename.endswith('.xml'):
            continue
        path = os.path.join(SCRIPTS_PATH, filename)
        tree = builder.build_from_file(path)
        render_node(tree, os.path.join(IMAGES_PATH, '{}.png'.format(filename.replace('.xml', ''))))
