import grpc from 'k6/experimental/grpc';
import {Counter, Trend} from 'k6/metrics';

const host = __ENV.HOST || '127.0.0.1:8080';

const streamCounter = new Counter('stream_counter');
const totalTime = new Trend('total_time', true);
const tokenizationTIme = new Trend('tokenization_time', true);
const queueTime = new Trend('queue_time', true);
const inferenceTime = new Trend('inference_time', true);

export const inputs = 'A path from a point approximately 330 metres east of the most south westerleasterly corner of Unit 4 Foundry Industrial Estate, then proceeding in a generally east-north-east direction for approximately 64 metres to a point approximately 282 metres east-south-east of the most easterly corner of Unit 2 Foundry Industrial Estate, Victoria Street, Widnes and approximately 259 metres east of the most southerly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 350 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road, Widnes, and approximately 47 metres west-south-west of the stub end of Cornubia Road be diverted to a 3 metre wide path from a point approximately 183 metres east-south-east of the most easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street and approximately 272 metres east of the most north-easterly corner of 26 Ann Street West, Widnes, then proceeding in a generally north easterly direction for approximately 58 metres to a point approximately 216 metres east-south-east of the most easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 221 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally easterly direction for approximately 45 metres to a point approximately 265 metres east-south-east of the most north-easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 265 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-south-east direction for approximately 102 metres to a point approximately 366 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 463 metres east of the most north easterly corner of 22 Ann Street West, Widnes, then proceeding in a generally north-north-easterly direction for approximately 19 metres to a point approximately 368 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 512 metres east of the most south easterly corner of 17 Batherton Close, Widnes then proceeding in a generally east-south, easterly direction for approximately 16 metres to a point approximately 420 metres east-south-east of the most southerly corner of Unit 2 Foundry';

export const options = {
    scenarios: {
        throughput: {
            executor: 'shared-iterations',
            vus: 1,
            iterations: 1,
            maxDuration: '2m',
            gracefulStop: '1s',
        },
    },
};


const client = new grpc.Client();

client.load([], '../proto/tei.proto');

export default function () {
    if (__ITER == 0) {
        client.connect(host, {
            plaintext: true
        });
    }

    const stream = new grpc.Stream(client, 'tei.v1.Embed/EmbedStream');

    stream.on('data', (res) => {
        totalTime.add(res.metadata.totalTimeNs / 1e6);
        tokenizationTIme.add(res.metadata.tokenizationTimeNs / 1e6);
        queueTime.add(res.metadata.queueTimeNs / 1e6);
        inferenceTime.add(res.metadata.inferenceTimeNs / 1e6);
    });

    stream.on('error', (err) => {
        console.log('Stream Error: ' + JSON.stringify(err));
    });

    const payload = {
        inputs: inputs,
        truncate: true,
    };

    // send 10000 requests
    for (let i = 0; i < 10000; i++) {
        stream.write(payload);
    }

    // close the client stream
    stream.end();
}
