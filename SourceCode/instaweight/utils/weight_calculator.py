

def using_hg(hg):
    # converting into cm
    hg = hg *  2.54
    return ((hg**2)+22)/100



def using_bl(hg, bl):
    weight = ((hg**2)*bl)/660
    return weight