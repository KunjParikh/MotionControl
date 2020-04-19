import pickle
import matplotlib.pyplot as plt
import params
df_state = pickle.load( open( "df_state.p", 'rb'))

params = params.Params()
for function in params.functions:
    x = df_state[function.name]
    plt.plot(x[:][0], 'r') # z_c
    plt.plot(x[:][1], 'g') # dz_c/dx
    plt.plot(x[:][2], 'b') # dz_c/dy
    plt.yscale("linear")
    plt.title(function.name)
    plt.show()

df_error = pickle.load( open( "df_error.p", 'rb'))

for function in params.functions:
    x = df_error[function.name]
    plt.plot(x[:][0], 'r') # z_c
    plt.plot(x[:][1], 'g') # dz_c/dx
    plt.plot(x[:][2], 'b') # dz_c/dy
    plt.yscale("linear")
    plt.title(function.name)
    plt.show()