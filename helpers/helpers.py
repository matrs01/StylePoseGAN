def cycle_dataloader(data_loader):
    """
    <a id="cycle_dataloader"></a>
    ## Cycle Data Loader
    Infinite loader that recycles the data loader after each epoch
    """
    while True:
        for batch in data_loader:
            yield batch
