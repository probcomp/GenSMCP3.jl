function my_mala(tr)
    new_x = random(ULA, (prev_x(tr), last_x(tr), tr[:steps => T(tr) => :yₜ])...)

    p_new_x = logpdf(ULA, new_x, (prev_x(tr), last_x(tr), tr[:steps => T(tr) => :yₜ])...)
    p_old_x = logpdf(ULA, last_x(tr), (prev_x(tr), new_x, tr[:steps => T(tr) => :yₜ])...)
    mh_proposal_ratio = p_new_x - p_old_x

    new_tr, model_ratio, _, _ = Gen.update(tr, choicemap((last_x_addr(tr), new_x)))

    # MH acceptance ratio
    mh_accept_ratio = mh_proposal_ratio + model_ratio

    # MH accept/reject
    if log(rand()) < mh_accept_ratio
        return new_tr
    else
        return tr
    end
end