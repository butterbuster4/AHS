def fitness_individual(tile_matrix, similarity_matrix_hori, similarity_matrix_vrti):
    """
        Calculates the fitness valuse of a tile group
    """
    value = 0
    # calculate the hori similarity
    hori_similarity = 0
    for row in tile_matrix:
        for i, tile in enumerate(row):
            if tile is not None and i < len(row) - 1:
                if row[i+1] is not None:
                    index_left = tile.index
                    index_right = row[i+1].index
                    hori_similarity += similarity_matrix_hori[index_left][index_right]
                    
    # calculate the vrti similarity
    vrti_similarity = 0
    for i, row in enumerate(tile_matrix):
        for j, tile in enumerate(row):
            if tile is not None and i < len(tile_matrix) - 1:
                if tile_matrix[i+1][j] is not None:
                    index_top = tile.index
                    index_bottom = tile_matrix[i+1][j].index
                    vrti_similarity += similarity_matrix_vrti[index_top][index_bottom]

    value = hori_similarity + vrti_similarity
    
    return value