from src.utilities.fitness.math_functions import pdiv, psqrt, plog
import numpy as np

def data_to_array(filename, delimiter=None):
    # Try to auto-detect the field separator (i.e. delimiter).
    if delimiter is None:
        f = open(filename)
        for line in f:
            if line.startswith("#") or len(line) < 2:
                # Skip excessively short lines or commented out lines.
                continue

            else:
                # Set the delimiter.
                if "\t" in line:
                    delimiter = "\t"
                    break
                elif "," in line:
                    delimiter = ","
                    break
                elif ";" in line:
                    delimiter = ";"
                    break
                elif ":" in line:
                    delimiter = ":"
                    break
                else:
                    print("utilities.fitness.get_data.get_Xy_train_test_separate\n"
                            "Warning: Dataset delimiter not found. "
                            "Defaulting to whitespace delimiter.")
                    delimiter = " "
                    break
        f.close()
    Xy = np.genfromtxt(filename, skip_header=1,delimiter=delimiter)
    return Xy

# Ejemplo de función que se quiere encontrar
def eval_indiv(x):
    x = np.transpose(x) # shape (dim, samples)
    result = np.sin(x[1])+plog(4*x[0])+5
    return result


if __name__ == '__main__':
    # Usaremos mismos inputs que se usaron para generar dataset Vladislavleva4 (ejemplo PonyEG2)

    # Funcion a aproximar: sin(x_1) + log(4*x_0) + 5
    test_filename = '../datasets/Vladislavleva4/Test.txt'
    test_Xy = data_to_array(test_filename)
    test_X = test_Xy[:,0:2] # Solo consideraremos 2 variables
    print(test_X.shape)
    train_filename = '../datasets/Vladislavleva4/Train.txt'
    train_Xy = data_to_array(train_filename)
    train_X = train_Xy[:,0:2] # Solo consideraremos 2 variables
    print(train_X.shape)

    teo_y = eval_indiv(train_X)
    train_Xy = np.concatenate((train_X, np.expand_dims(teo_y, 1)), 1)
    np.savetxt('/home/franrosi/Escritorio/Proyecto CompEv/PonyGE2-master/datasets/Custom_fran/Train.txt', train_Xy, delimiter='\t', fmt='%.11f')

    teo_y = eval_indiv(test_X)
    test_Xy = np.concatenate((test_X, np.expand_dims(teo_y, 1)), 1)
    np.savetxt('/home/franrosi/Escritorio/Proyecto CompEv/PonyGE2-master/datasets/Custom_fran/Test.txt', test_Xy, delimiter='\t', fmt='%.11f')
