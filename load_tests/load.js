import {check} from 'k6';
import http from 'k6/http';
import {Trend} from 'k6/metrics';

const host = __ENV.HOST || '127.0.0.1:3000';

const totalTime = new Trend('total_time', true);
const tokenizationTIme = new Trend('tokenization_time', true);
const queueTime = new Trend('queue_time', true);
const inferenceTime = new Trend('inference_time', true);

export const inputs = 'A path from a point approximately 330 metres east of the most south westerleasterly corner of Unit 4 Foundry Industrial Estate, then proceeding in a generally east-north-east direction for approximately 64 metres to a point approximately 282 metres east-south-east of the most easterly corner of Unit 2 Foundry Industrial Estate, Victoria Street, Widnes and approximately 259 metres east of the most southerly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 350 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road, Widnes, and approximately 47 metres west-south-west of the stub end of Cornubia Road be diverted to a 3 metre wide path from a point approximately 183 metres east-south-east of the most easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street and approximately 272 metres east of the most north-easterly corner of 26 Ann Street West, Widnes, then proceeding in a generally north easterly direction for approximately 58 metres to a point approximately 216 metres east-south-east of the most easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 221 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally easterly direction for approximately 45 metres to a point approximately 265 metres east-south-east of the most north-easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 265 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-south-east direction for approximately 102 metres to a point approximately 366 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 463 metres east of the most north easterly corner of 22 Ann Street West, Widnes, then proceeding in a generally north-north-easterly direction for approximately 19 metres to a point approximately 368 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 512 metres east of the most south easterly corner of 17 Batherton Close, Widnes then proceeding in a generally east-south, easterly direction for approximately 16 metres to a point approximately 420 metres east-south-east of the most southerly corner of Unit 2 Foundry';

export const options = {
    thresholds: {
        http_req_failed: ['rate==0'],
    },
    scenarios: {
        // throughput: {
        //     executor: 'shared-iterations',
        //     vus: 5000,
        //     iterations: 5000,
        //     maxDuration: '2m',
        //     gracefulStop: '1s',
        // },
        load_test: {
            executor: 'constant-arrival-rate',
            duration: '30s',
            preAllocatedVUs: 5000,
            rate: 50,
            timeUnit: '1s',
            gracefulStop: '1s',
        },
    },
};

export default function () {
    const payload = JSON.stringify({
        inputs: inputs,
        // query: inputs,
        // texts: [inputs],
        truncate: true,
    });

    const headers = {'Content-Type': 'application/json'};
    const res = http.post(`http://${host}/`, payload, {
        headers, timeout: '20m'
    });

    check(res, {
        'Post status is 200': (r) => res.status === 200,
    });

    if (res.status === 200) {
        totalTime.add(res.headers["X-Total-Time"]);
        tokenizationTIme.add(res.headers["X-Tokenization-Time"]);
        queueTime.add(res.headers["X-Queue-Time"]);
        inferenceTime.add(res.headers["X-Inference-Time"]);
    } else {
        console.log(res.error);
    }
}
