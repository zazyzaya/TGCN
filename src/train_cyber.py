import torch
from torch.optim import Adam

import generators as g
from models.serial_model import SerialTGCN

LR=0.01
def train_cyber(data, epochs=10000, te_history=0):
    # Leave all params as default for now
    SKIP = data.te_starts 
    model = SerialTGCN(
        data.x.size(1), 32, 16 
    )
    opt = Adam(model.parameters(), lr=LR)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()

        # Get embeddings
        zs = model(data.x, data.eis[:SKIP], data.all)
        
        # Generate positive and negative samples from this and the next time step
        p,n,z = g.link_prediction(data, data.tr, zs, include_tr=False, end=SKIP)
        dp,dn,dz = g.dynamic_link_prediction(data, data.tr, zs, include_tr=False, end=SKIP)
        
        loss = model.loss_fn(p+dp, n+dn, torch.cat([z, dz], dim=0))
            

        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item() 

        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis, data.tr)[SKIP-1:]
        
            if not dynamic:
                p,n,z = g.link_prediction(data, data.va, zs, start=SKIP)
                st, sf = model.score_fn(p,n,z)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores) )
                )

                avg = sscores[1]

            else:
                # VGRNN is providing priors, which are built from the previous timestep
                # already, thus there is no need to shift the selected ei's as the 
                # dynamic functions do 
                if model.__class__ == VGRNN:
                    zs = zs[1:]
                    dp,dn,dz = g.link_prediction(data, None, zs, start=SKIP)
                else:
                    dp,dn,dz = g.dynamic_link_prediction(data, None, zs, start=SKIP-1)
                
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.dynamic_new_link_prediction(data, None, zs, start=SKIP-1)
                if model.__class__ == VGRNN:
                    dz = zs # Again, we don't need to shift the VGRNN embeds backward
                
                dt, df = model.score_fn(dp,dn,dz)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tDet %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores) )
                )

                avg = (
                    dscores[1] +
                    dnscores[1]
                )
            
            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
            else:
                # Though it's not reflected in the code, the authors for VGRNN imply in the
                # supplimental material that after 500 epochs, early stopping may kick in 
                if e > 500:
                    no_improvement += 1
                if no_improvement > PATIENCE:
                    print("Early stopping...\n")
                    break


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