def train_cyber(data, epochs=10000, te_history=0):
    # Leave all params as default for now
    model = SerialTGCNGraphGRU(
        data.x.size(1),
        gcn_out_dim=8,
        gru_hidden_units=4
    )
    opt = Adam(model.parameters(), lr=LR)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()

        ei_tr = data.tr_slice()
        ei_te = data.te_slice()

        # Get embedding
        zs = model(data.x, ei_tr, data.tr)
        
        # Calculate static and dynamic loss
        p,n,z = g.link_prediction(data, data.tr, zs, trange=data.tr_range, include_tr=False)
        dp,dn,dz = g.dynamic_link_prediction(data, data.tr, zs, trange=data.tr_range, include_tr=False)
        dnp, dnn, dnz = g.dynamic_new_link_prediction(data, data.tr, zs, trange=data.tr_range, include_tr=False)

        loss = model.loss_fn(
            p+dp+dnp, n+dn+dnn, torch.cat([z, dz, dnz], dim=0)
        )
        loss.backward()
        opt.step()

        trloss = loss.item() 

        with torch.no_grad():
            model.eval()
            zs = model(data.x, ei_tr, data.tr)
        
            p,n,z = g.link_prediction(data, data.va, zs, trange=data.tr_range)
            st, sf = model.score_fn(p[SKIP:],n[SKIP:],z[SKIP:])
            sscores = get_score(st, sf)

            dp,dn,dz = g.dynamic_link_prediction(data, data.va, zs, trange=data.tr_range)
            dt, df = model.score_fn(dp[SKIP:],dn[SKIP:],dz[SKIP:])
            dscores = get_score(dt, df)

            dp,dn,dz = g.dynamic_new_link_prediction(data, data.va, zs, trange=data.tr_range)
            dt, df = model.score_fn(dp[SKIP:],dn[SKIP:],dz[SKIP:])
            dnscores = get_score(dt, df)

            print(
                '[%d] Loss: %0.4f  \n\tSt %s  \n\tDy %s  \n\tDyN %s' %
                (e, trloss, fmt_score(sscores), fmt_score(dscores), fmt_score(dnscores) )
            )

            avg = (
                sscores[0] + sscores[1] + 
                dscores[0] + dscores[1] +
                dnscores[0] + dnscores[1]
            )
            
            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > PATIENCE:
                    print("Early stopping...\n")
                    break


    model = best[1]
    zs = None
    with torch.no_grad():
        model.eval()
        z_t = torch.cat([
            model.encode(data.x, ei_tr[te_history:], data.tr),
            model.encode(data.x, ei_te, data.te)
        ], dim=0) 

        zs = model.recurrent(z_t)[data.te_starts-te_history:]

    # Scores all edges and matches them with name/timestamp
    edges = []
    for i in range(zs.size(0)):
        idx = i + data.te_starts

        ei = data.eis[idx]
        scores = model.decode(ei[0], ei[1], zs[i])
        names = data.format_edgelist(idx)

        for i in range(len(names)):
            edges.append(
                (scores[i].item(), names[i])
            )

    edges.sort(key=lambda x : x[0], reverse=True)
    for e in edges:    