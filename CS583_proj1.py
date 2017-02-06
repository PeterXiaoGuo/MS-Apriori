import numpy as np
import scipy.sparse as sp
import pandas as pd
import sys

def MSApriori(file_args, file_data):
    
    def input_args(file_args):
        ms = dict()
        sdc = 1
        x_cannot = []
        x_must = []
        for i in open(file_args, "r"):
            i = i.rstrip("\n")
            if i.startswith("MIS"):
                j = i.split(" = ")
                ms.update({j[0][4:-1]: float(j[1])})
            elif i.startswith("SDC"):
                sdc = float(i.split("=")[1])
            elif i.startswith("cannot_be_together"):
                x_cannot = [j.split(", ") for j in i.split(": ")[1][1:-1].split("}, {")]
            elif i.startswith("must"):
                x_must = [j for j in i.split(": ")[1].split(" or ")]
        op = {"ms": ms, "sdc": sdc, "x_cannot": x_cannot, "x_must": x_must}
        return(op)

    def input_data(file_data, columns):
        s = pd.read_csv(file_data, header = None, sep = "\t",squeeze = True)
        op = s.str[1:-1].str.get_dummies(sep = ", ").reindex(columns = columns, fill_value = 0)
        return(op)

    def sup(xL):
        op = np.mean([X[:, i].all(axis = 1) for i in xL], axis = 1)
        return(op)

    def pair_sup_mis(x):
        x_t = x[:, np.newaxis]
        x_sup = sup(x_t)
        x_mis = [ms_dict[i] for i in x]
        x_sup_t = x_sup[:, np.newaxis]
        iL = sp.coo_matrix(np.triu((x_sup_t >= x_mis).T & (np.abs(x_sup_t - x_sup) < args["sdc"]), 1)).nonzero()
        op = list(zip(x[iL[0]], x[iL[1]]))
        return(op)

    def frequent(xL):
        x_sup = sup(xL)
        sup_dict.update(dict(zip(xL, x_sup)))
        op = [xL[j] for j in np.where(x_sup >= [ms_dict[i[0]] for i in xL])[0]]
        xL_dropfirst = set(tuple(i[1:]) for i in xL)
        sup_dict.update(dict(zip(xL_dropfirst, sup(xL_dropfirst))))
        return(op)

    def append_set(xL, x_base):
        if len(xL):
            op = [tuple(i) for i in np.hstack([np.tile(x_base, (len(xL), 1)), xL])]
        else:
            op = []
        return(op)

    def prune_candidate(xL):
        if xL:
            op = [xL[l] for l in np.where(np.all([[any(set(k).issubset(j) for k in F[-1]) for j in np.delete(xL, i, axis = 1)] for i in range(1, len(xL[0]))], axis = 0))[0]]
        else:
            op = []
        return(op)
    
    def output_frequent(F):
        op = []
        for i, ival in enumerate(F):
            if ival:
                op.append("Frequent {}-itemsets\n".format(i+1))
                if i == 0:
                    op += ["\t{} : {}".format(int(sup_dict[j]*len(X)), {int(id_dict[k]) for k in j}) for j in ival]
                else:
                    op += ["\t{} : {}\nTailcount = {}".format(int(sup_dict[j]*len(X)), {int(id_dict[k]) for k in j}, int(sup_dict[j[1:]]*len(X))) for j in ival]
                op.append("\nTotal number of frequent {}-itemsets = {}\n\n".format(i+1, len(ival)))
        print("\n".join(op))
        
    # Read Arguments and Transaction Data
    args = input_args(file_args)
    ms = pd.Series(args["ms"], name = "MIS").sort_values().reset_index()
    id_dict = ms["index"].to_dict()
    ms_dict = ms["MIS"].to_dict()
    id_dict_inv = {val: key for key, val in id_dict.items()}
    x_must = [id_dict_inv[i] for i in args["x_must"]]
    x_cannot = [tuple(np.sort([id_dict_inv[j] for j in i])) for i in args["x_cannot"]]
    X = input_data(file_data, ms["index"]).values
    
    ## Level 1
    I = [(i,) for i, ival in enumerate(ms_dict)]
    Isup = sup(I)
    sup_dict = dict(zip(I, Isup))
    Li = (Isup > ms["MIS"]).argmax()
    L = [i for i in range(Li, ms.shape[0]) if Isup[i] > ms["MIS"][Li]]
    F = [[(i,) for i in np.where(Isup > ms["MIS"])[0]]]
    
    ## Level >= 2
    C = [i for i in pair_sup_mis(np.array(L)) if i[0] in np.array(F[0]).T[0]]
    while C:
        F.append(frequent(C))
        Ls = pd.DataFrame(F[-1])
        C = sum([append_set(pair_sup_mis(group.values), name) for name, group in Ls.groupby(list(range(len(F)-1)))[len(F)-1]], [])
        C = prune_candidate(C)
        
    ## Prune and Output
    F_prune = [[j for j in i if any(k in j for k in x_must) & ~any(set(k).issubset(j) for k in x_cannot)] for i in F]
    output_frequent(F_prune)

MSApriori(*sys.argv[1:])
