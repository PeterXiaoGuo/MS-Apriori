import numpy as np
import scipy.sparse as sp
import pandas as pd
import sys

def MSApriori(file_args, file_data):
    
    def input_args(file_args):
        ms_i = []
        ms_val = []
        sdc = 1
        x_cannot = []
        x_must = []
        for i in open(file_args, "r"):
            i = i.rstrip("\n")
            if i.startswith("MIS"):
                j = i.split(" = ")
                ms_i.append(j[0][4:-1])
                ms_val.append(float(j[1]))
            elif i.startswith("SDC"):
                sdc = float(i.split("=")[1])
            elif i.startswith("cannot_be_together"):
                x_cannot = [j.split(", ") for j in i.split(": ")[1][1:-1].split("}, {")]
            elif i.startswith("must"):
                x_must = [j for j in i.split(": ")[1].split(" or ")]
        ms_sort = np.argsort(ms_val)
        op = {"ms_i": np.array(ms_i)[ms_sort], "ms_val": np.array(ms_val)[ms_sort], "sdc": sdc, "x_cannot": x_cannot, "x_must": x_must}
        return(op)

    def input_data(file_data, columns):
        s = pd.read_csv(file_data, header = None, sep = "\t",squeeze = True)
        op = s.str.strip("{} ").str.get_dummies(sep = ", ").reindex(columns = columns, fill_value = 0)
        return(op)

    def sup(xL):
        op = np.mean([X[:, i].all(axis = 1) for i in xL], axis = 1)
        return(op)

    def pair_sup_mis(x):
        x_t = x[:, np.newaxis]
        x_sup = sup(x_t)
        x_mis = [args["ms_val"][i] for i in x]
        x_sup_t = x_sup[:, np.newaxis]
        iL = sp.coo_matrix(np.triu((x_sup_t >= x_mis).T & (np.abs(x_sup_t - x_sup) <= args["sdc"]), 1)).nonzero()
        op = list(zip(x[iL[0]], x[iL[1]]))
        return(op)
    
    def frequent(xL):
        op = []
        if xL:
            x_sup = sup(xL)
            sup_dict.update(dict(zip(xL, x_sup)))
            op += [xL[j] for j in np.where(x_sup >= [args["ms_val"][i[0]] for i in xL])[0]]
            xL_dropfirst = set(tuple(i[1:]) for i in xL)
            sup_dict.update(dict(zip(xL_dropfirst, sup(xL_dropfirst))))
        return(op)
    
    def append_set(xL, x_base):
        op = []
        if len(xL):
            op += [tuple(i) for i in np.hstack([np.tile(x_base, (len(xL), 1)), xL])]
        return(op)
    
    def prune_candidate(xL):
        op = []
        if xL:
            op += [xL[l] for l in np.where(np.all([[any(set(k).issubset(j) for k in F[-1]) for j in np.delete(xL, i, axis = 1)] for i in range(1, len(xL[0]))], axis = 0))[0]]
        return(op)
    
    def output_frequent(F):
        op = []
        ms_i_int = args["ms_i"].astype("int")
        for i, ival in enumerate(F):
            if ival:
                op.append("Frequent {}-itemsets\n".format(i+1))
                if i == 0:
                    op += ["\t{} : {}".format(int(sup_dict[j]*len(X)), {ms_i_int[j]}) for j in ival]
                else:
                    op += ["\t{} : {}\nTailcount = {}".format(int(sup_dict[j]*len(X)), list(ms_i_int[[j]]), int(sup_dict[j[1:]]*len(X))) for j in ival]
                op.append("\nTotal number of frequent {}-itemsets = {}\n\n".format(i+1, len(ival)))
        print("\n".join(op).replace("[", "{").replace("]", "}"))
        
    # Read Arguments and Transaction Data
    args = input_args(file_args)
    id_dict = {i[1]: i[0] for i in enumerate(args["ms_i"])}
    x_must = [id_dict[i] for i in args["x_must"]]
    x_cannot = [tuple(np.sort([id_dict[j] for j in i])) for i in args["x_cannot"]]
    X = input_data(file_data, args["ms_i"]).values
    
    ## Level 1
    I = [(i,) for i, ival in enumerate(args["ms_val"])]
    Isup = sup(I)
    sup_dict = dict(zip(I, Isup))
    Li = (Isup >= args["ms_val"]).argmax()
    L = [i for i in range(Li, len(args["ms_val"])) if Isup[i] >= args["ms_val"][Li]]
    F = [[(i,) for i in np.where(Isup >= args["ms_val"])[0]]]
    
    ## Level >= 2
    C = [i for i in pair_sup_mis(np.array(L)) if i[0] in np.array(F[0]).T[0]]
    F_last = frequent(C)
    while F_last:
        F.append(F_last)
        Ls = pd.DataFrame(F[-1])
        C = sum([append_set(pair_sup_mis(group.values), name) for name, group in Ls.groupby(list(range(len(F)-1)))[len(F)-1]], [])
        C = prune_candidate(C)
        F_last = frequent(C)
        
    ## Prune and Output
    F_prune = [[j for j in i if any(k in j for k in x_must) & ~any(set(k).issubset(j) for k in x_cannot)] for i in F]
    output_frequent(F_prune)

MSApriori(*sys.argv[1:])