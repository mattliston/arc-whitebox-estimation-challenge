"""Inert fixed physical all-distinct cubic sampling-variance pilot.

This is not the failed Mehler/tangent law and not a deployable estimator.
All physical paths, unbiased batch queries and uncertainty arithmetic use FNP.
"""
import hashlib

WIDTH, BATCH, BATCHES, STREAMS = 1024, 1024, 16, 2
SEEDS = (202609131101, 202609131102)
PRECISION = 1e-9
CAP = 2_000_000_000_000
DENSE_ONE = 2 * BATCH * WIDTH * (2 * WIDTH - 1)
VECTOR_ONE = 2 * WIDTH * (2 * WIDTH - 1)
BATCH_NAMES = tuple('stream_' + str(r) + '_batch_' + str(k).zfill(2)
                    for k in range(BATCHES) for r in range(STREAMS))
SECTIONS = ('ingress', 'fixed_tiny_identity', 'query_weights', *BATCH_NAMES,
            'matched_variance_and_exports')
SECTION_CAPS = (250_000_000, 4_000_000, 20_000_000,
                *(49_000_000_000 for name in BATCH_NAMES), 20_000_000)
PRACTICAL_SAMPLES = 262144
PLANNING_MULTIPLIER = 2


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def cost_certificate():
    require(DENSE_ONE == 4292870144, 'Fixed full FP64 batched matrix tariff')
    require(sum(SECTION_CAPS) == 1568294000000 and sum(SECTION_CAPS) < CAP,
            'Fixed complete source allowances before measurement')
    return dict(streams=STREAMS, batches_per_stream=BATCHES, rows_per_batch=BATCH,
        physical_samples=STREAMS*BATCHES*BATCH, width=WIDTH, prefix_layers=9,
        dense_FP64_products_per_batch=11, dense_FP64_one=DENSE_ONE,
        dense_FP64_total=STREAMS*BATCHES*11*DENSE_ONE,
        vector_FP64_products_per_batch=1, vector_FP64_one=VECTOR_ONE,
        section_caps=dict(zip(SECTIONS, SECTION_CAPS)),
        complete_source_allowance=sum(SECTION_CAPS), hard_cap=CAP,
        physical_and_query_dtype='float64', stored_weight_dtype='float32',
        no_alive_mask=True, tangent_law=False, all_distinct_tensor_materialized=False,
        variance_only=True, whole_estimator_cost=False,
        practical_sample_ceiling=PRACTICAL_SAMPLES,
        planning_multiplier=PLANNING_MULTIPLIER,
        planning_multiplier_is_confidence_bound=False)


def run(fnp, load_serialized, section, own, ctx):
    def array(value):
        return own(value)
    def scalar(value):
        return array(fnp.full((1,), value, dtype=fnp.float64))
    def finite(value):
        require(bool(fnp.all(fnp.isfinite(value))), 'Actual paid array finite')
    def export(value):
        copied=array(value.astype(fnp.float64))
        finite(copied)
        raw, shape, dtype=copied._fetch_data()
        require(tuple(shape)==tuple(value.shape) and str(dtype)=='float64'
            and len(raw)==8*value.size, 'Detached paid FP64 snapshot')
        return dict(shape=list(shape), dtype='float64', bytes=len(raw),
                    raw_sha256=hashlib.sha256(raw).hexdigest())
    def ingress():
        raw=load_serialized()
        weights=array(fnp.frombuffer(raw['weights'],dtype=fnp.float32)
            .reshape((10,WIDTH,WIDTH)).astype(fnp.float64))
        response={}
        for name in ('alpha','pdf','sigma'):
            response[name]=array(fnp.frombuffer(raw[name],dtype=fnp.float64)
                .reshape((WIDTH,)).astype(fnp.float64))
            finite(response[name])
        finite(weights)
        zero, one, two, three, six = (scalar(x) for x in (0.,1.,2.,3.,6.))
        require(bool(fnp.all(response['sigma']>zero))
            and bool(fnp.all(response['pdf']>=zero)), 'Current pre9 response domain')
        sensitivity=array(-response['alpha']*response['pdf']/
                          (six*response['sigma']*response['sigma']))
        finite(sensitivity)
        return weights,sensitivity,(zero,one,two,three,six)
    weights,sensitivity,(zero,one,two,three,six)=section('ingress',ingress)

    def query(h,q,q2,q3):
        b=int(h.shape[0])
        count=scalar(float(b))
        factor=scalar(float(b*b))/scalar(float((b-1)*(b-2)))
        x=array(h-fnp.sum(h,axis=0)/count)
        x2=array(x*x)
        u=array(x@q)
        v=array(x2@q2)
        cubic=array(fnp.sum(u*u*u-three*u*v,axis=0)/count)
        diagonal=array((fnp.sum(x2*x,axis=0)/count)@q3)
        return array(factor*(cubic+two*diagonal))

    def identity():
        # Independent direct 3-index sum on one fixed 8x4 fixture, not 1024^3.
        h=array(fnp.sin(fnp.arange(32,dtype=fnp.float64)).reshape((8,4)))
        q=array(fnp.cos(fnp.arange(12,dtype=fnp.float64)).reshape((4,3)))
        q2=array(q*q);q3=array(q2*q)
        actual=query(h,q,q2,q3)
        center=array(h-fnp.sum(h,axis=0)/scalar(8.))
        literal=array(fnp.zeros((3,),dtype=fnp.float64))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if i!=j and i!=k and j!=k:
                        moment=fnp.sum(center[:,i]*center[:,j]*center[:,k])/scalar(8.)
                        literal=array(literal+moment*q[i]*q[j]*q[k])
        literal=array(literal*(scalar(64.)/scalar(42.)))
        error=fnp.max(fnp.abs(actual-literal))
        shift=array(fnp.arange(4,dtype=fnp.float64))
        translated=query(array(h+shift),q,q2,q3)
        translation_error=fnp.max(fnp.abs(translated-actual))
        finite(actual);finite(literal);finite(translated)
        require(bool(error<=scalar(1e-12)) and bool(translation_error<=scalar(1e-12)),
            'Fixed direct-distinct and translation identities before sampling')
        return dict(maxabs_direct=float(error),maxabs_translation=float(translation_error),
            threshold=1e-12,dimension=4,rows=8,columns=3,direct_distinct_terms=24,
            passed=True)
    tiny=section('fixed_tiny_identity',identity)

    def query_weights():
        q=weights[9]
        q2=array(q*q);q3=array(q2*q)
        finite(q2);finite(q3)
        generators=tuple(fnp.random.default_rng(seed) for seed in SEEDS)
        return q,q2,q3,generators
    q,q2,q3,generators=section('query_weights',query_weights)
    batches=[[],[]]
    counts=dict(physical=0,query=0,vector=0)
    def dense(left,right,kind):
        require(tuple(left.shape)==(BATCH,WIDTH)
            and tuple(right.shape)==(WIDTH,WIDTH)
            and str(left.dtype)==str(right.dtype)=='float64', 'Fixed physical/query FP64 shapes')
        before=int(ctx.flops_used)
        result=array(left@right)
        require(int(ctx.flops_used)-before==DENSE_ONE, 'Actual exact complete dense tariff')
        counts[kind]+=1
        return result
    def batch_statistic(generator):
        h=array(generator.standard_normal((BATCH,WIDTH)))
        finite(h)
        for layer in range(9):
            h=array(fnp.maximum(dense(h,weights[layer],'physical'),zero))
            finite(h)
        count=scalar(float(BATCH))
        factor=scalar(float(BATCH*BATCH))/scalar(float((BATCH-1)*(BATCH-2)))
        x=array(h-fnp.sum(h,axis=0)/count)
        x2=array(x*x)
        u=dense(x,q,'query');v=dense(x2,q2,'query')
        cubic=array(fnp.sum(u*u*u-three*u*v,axis=0)/count)
        diag_input=array(fnp.sum(x2*x,axis=0)/count)
        before=int(ctx.flops_used)
        diagonal=array(diag_input@q3)
        require(int(ctx.flops_used)-before==VECTOR_ONE, 'Actual complete vector query tariff')
        counts['vector']+=1
        result=array(sensitivity*factor*(cubic+two*diagonal))
        finite(result)
        return result
    for k in range(BATCHES):
        for r,generator in enumerate(generators):
            name=BATCH_NAMES[k*STREAMS+r]
            batches[r].append(section(name,lambda:batch_statistic(generator)))
    del generator,generators

    def statistics():
        n=scalar(float(WIDTH));k=scalar(float(BATCHES))
        batch_arrays=tuple(array(fnp.stack(values,axis=0)) for values in batches)
        def variance_trace(values):
            rows=int(values.shape[0])
            mean=array(fnp.sum(values,axis=0)/scalar(float(rows)))
            centered=array(values-mean)
            return fnp.sum(centered*centered)/(scalar(float(rows-1))*n)
        # Each value below is tr(sample covariance)/n, not diagonalized units.
        stream_q=tuple(variance_trace(value) for value in batch_arrays)
        half_q=tuple(variance_trace(value[start:start+8])
            for value in batch_arrays for start in (0,8))
        for value in (*stream_q,*half_q):
            finite(value)
        pooled=(stream_q[0]+stream_q[1])/(scalar(4.)*k)
        qhalf=array(fnp.stack(half_q))
        low=fnp.min(qhalf);high=fnp.max(qhalf)
        stream_low=fnp.minimum(stream_q[0],stream_q[1])
        stream_high=fnp.maximum(stream_q[0],stream_q[1])
        positive=bool(low>zero) and bool(stream_low>zero)
        half_consistency=bool(high<=scalar(4.)*low)
        stream_consistency=bool(stream_high<=two*stream_low)
        # Empirical conservative planning rule, explicitly NOT a confidence bound.
        qguard=scalar(float(PLANNING_MULTIPLIER))*fnp.maximum(high,stream_high)
        planned_noise=qguard*scalar(float(BATCH))/scalar(float(PRACTICAL_SAMPLES))
        projected_samples=qguard*scalar(float(BATCH))/scalar(PRECISION)
        required_blocks=fnp.maximum(one,fnp.ceil(projected_samples/scalar(32768.)))
        affordable=bool(required_blocks<=scalar(8.))
        finite(pooled);finite(planned_noise);finite(projected_samples)
        passed=positive and half_consistency and stream_consistency and affordable
        primary=dict(pooled_host_MSE_variance=float(pooled),
            stream_trace_covariance_per_unit=[float(value) for value in stream_q],
            half_trace_covariance_per_unit=[float(value) for value in half_q],
            planning_multiplier=PLANNING_MULTIPLIER,positive_variance=positive,
            stream_ratio_limit=2.,half_ratio_limit=4.,
            stream_consistency_passed=stream_consistency,half_consistency_passed=half_consistency,
            projected_noise_at_sample_ceiling=float(planned_noise),
            projected_required_samples=float(projected_samples),
            projected_required_pilot_blocks=int(required_blocks[0]),
            practical_sample_ceiling=PRACTICAL_SAMPLES,host_MSE_variance_target=PRECISION,
            variance_planning_passed=passed,confidence_bound_claimed=False,
            full_scope_projection_decision_pending=True,
            projected_precision_not_measured=True)
        return dict(name='fixed_physical_distinct_cubic_variance',
            primary=primary,tiny_identity=tiny,
            streams=STREAMS,batches_per_stream=BATCHES,rows_per_batch=BATCH,
            physical_samples=STREAMS*BATCHES*BATCH,seeds=list(SEEDS),
            matmul_counts=dict(counts),dense_matmul_flops=32*11*DENSE_ONE,
            vector_matmul_flops=32*VECTOR_ONE,
            exports=dict(stream0=export(batch_arrays[0]),stream1=export(batch_arrays[1])),
            physical_prefix_forward_batches=32,hard_pathwise_ReLU_layers=9,
            actual_law_is_unmasked=True,tangents_used=False,
            batch_k_statistic_unbiased_in_exact_arithmetic=True,
            variance_measured=True,signed_current_tensor_error_measured=False,
            raw_mean_or_final_score_measured=False)
    row=section('matched_variance_and_exports',statistics)
    require(counts==dict(physical=288,query=64,vector=32),
        'Exactly32 physical prefixes,64 dense and32 vector queries')
    return [row]
