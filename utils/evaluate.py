import numpy as np


def compute_ap(ranks, nres):
    """
    Compute average precision for given ranked indexes.

    Arguments
    ---------
    ranks:zerro-based ranks of positive images
    nres:number of positive images
    Returns
    --------
    ap:average precision
    """
    num_images = len(ranks)

    ap = 0.0

    recall_step = 1.0 / nres

    for index in range(num_images):
        rank = ranks[index]

        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(index) / rank

        precision_1 = float(index + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.0

    return ap


def compute_map(ranks, gnd, keeps=None):
    """
    Compute the mAP for a given set of returned reshults.

        Usgae:
        map = compute_map(ranks,gnd)
        computes mean average precision (map) only

        map,aps,pr,prs = compute_map(ranks,gnd,keeps)
        computes mean average precision (map), average precision (aps) for each query
        computes mean average precision at keeps (pr),precision at keeps(prs) for each query

        Notes:
         1) ranks starts from 0, keeps starts from 1, ranks.shape = db_size,query_nums
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """
    if keeps:
        mAP = 0.0
        query_nums = len(gnd)
        aps = np.zeros(query_nums)
        pr = np.zeros(len(keeps))
        prs = np.zeros((query_nums, len(keeps)))
        empty_num = 0

        for i in range(query_nums):
            query_gnd_ok = np.array(gnd[i]['ok'])

            if query_gnd_ok.shape[0] == 0:
                aps[i] = float('+inf')
                prs[i, :] = float('+inf')
                empty_num += 1
            else:
                try:
                    query_gnd_junk = np.array(gnd[i]['junk'])
                except:
                    query_gnd_junk = np.empty(0)

                pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_ok)]
                junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_junk)]

                num = 0
                index = 0
                if len(junk):
                    ip = 0
                    while ip < len(pos):
                        while index < len(junk) and pos[ip] > junk[index]:
                            num += 1
                            index += 1
                        pos[ip] -= num
                        ip += 1

                # compute ap
                ap = compute_ap(pos, len(query_gnd_ok))
                mAP += ap
                aps[i] = ap

                # compute precision @ k
                pos += 1
                for k in range(len(keeps)):
                    kp = min(max(pos), keeps[k])
                    prs[i, k] = (pos <= kp).sum() / kp
                pr += prs[i, :]

        mAP = mAP / (query_nums - empty_num)
        pr = pr / (query_nums - empty_num)
        return mAP, aps, pr, prs
    else:
        mAP = 0.0
        query_nums = len(gnd)
        aps = np.zeros(query_nums)
        empty_num = 0

        for i in range(query_nums):
            query_gnd_ok = np.array(gnd[i]['ok'])

            if query_gnd_ok.shape[0] == 0:
                aps[i] = float('+inf')
                empty_num += 1
            else:
                try:
                    query_gnd_junk = np.array(gnd[i]['junk'])
                except:
                    query_gnd_junk = np.empty(0)

                pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_ok)]
                junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_junk)]

                num = 0
                index = 0
                if len(junk):
                    ip = 0
                    while ip < len(pos):
                        while index < len(junk) and pos[ip] > junk[index]:
                            num += 1
                            index += 1
                        pos[ip] -= num
                        ip += 1

                # compute ap
                ap = compute_ap(pos, len(query_gnd_ok))
                mAP += ap
                aps[i] = ap

        mAP = mAP / (query_nums - empty_num)
        return mAP, aps


def compute_map_and_print(dataset, featuretype, mode, ranks, gnd, kappas=[1, 5, 10], verbose=False):

    # old evaluation protocol
    if dataset.startswith('oxford5k') or dataset.startswith('paris6k'):
        map, aps, _, _ = compute_map(ranks, gnd)
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map * 100, decimals=2)))

    # new evaluation protocol
    elif dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

        print('>> Test Dataset: {} *** Feature Type: {} >>'.format(dataset, featuretype))
        print('>> mAP Eeay: {}, Medium: {}, Hard: {}'.format(np.around(mapE * 100, decimals=2), np.around(mapM * 100, decimals=2), np.around(mapH * 100, decimals=2)))
        print('>> mP@k{} Easy: {}, Medium: {}, Hard: {}'.format(kappas, np.around(mprE * 100, decimals=2), np.around(mprM * 100, decimals=2), np.around(mprH * 100, decimals=2)))

        if verbose:
            print('>> Query aps: >>\nEeay: {}\nMedium: {}\nHard: {}'.format(np.around(apsE * 100, decimals=2), np.around(apsM * 100, decimals=2), np.around(apsH * 100, decimals=2)))

        return np.around(mapE * 100, decimals=2), np.around(mapM * 100, decimals=2), np.around(mapH * 100, decimals=2)
