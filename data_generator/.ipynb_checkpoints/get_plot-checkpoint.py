
def plot_array(arr, title=None):
    '''
    function for plotting the array 
    '''
    try:
        get_ipython().run_line_magic('matplotlib', 'ipympl')
    except:
        pass
    fig = plt.figure()
    plt.imshow(arr, cmap = 'gray', origin = 'lower')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    return fig
    