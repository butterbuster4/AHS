import numpy as np

from PIL import Image
from image_analysis import ImageAnalysis
from fitness import fitness_individual
from utils import create_image_from_matrix
import utils


class Individual(object):
    FITNESS_FACTOR = 1000
    PUZZLE_PIECE_NUMBER = 9

    def __init__(self, groups=None, matrixs=None):
        self.matrixs = utils.create_matrixs_from_groups_of_tiles(groups) if matrixs is None else matrixs
        self.groups = utils.create_groups_from_matrixs(self.matrixs) if groups is None else groups
        self._fitness = None
        
    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = 0
            for matrix in self.matrixs:
                fitness = fitness_individual(matrix, ImageAnalysis.similarity_matrix_hori, ImageAnalysis.similarity_matrix_vrti)
                self._fitness += fitness
        return self._fitness

    def to_image(self):
        """Converts individual to a showable image by assembling images of each group into one single image (no overlap)."""

        matrix_images = []
        for matrix in self.matrixs:
            # Step 1: convert each group (list of tiles) into one image
            tile_matrix = matrix
            puzzle_image = create_image_from_matrix(tile_matrix)
            matrix_images.append(puzzle_image)

        # Step 2: determine max height and total width
        cols = 5
        rows = (len(matrix_images) + cols - 1) // cols
        max_w = max(img.width for img in matrix_images)
        max_h = max(img.height for img in matrix_images)

        # Step 3: create a blank canvas for the final image
        final_image = Image.new('RGB', (cols * max_w, rows * max_h), (255, 255, 255))
        for idx, img in enumerate(matrix_images):
            row, col = divmod(idx, cols)
            final_image.paste(img, (col * max_w, row * max_h))

        return final_image
            