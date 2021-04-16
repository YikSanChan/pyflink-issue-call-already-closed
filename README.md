# PyFlink issue: call already closed

I run into an issue where a PyFlink job may end up with 3 very different outcomes, given very slight difference in input, and luck :(

The PyFlink job is simple. It first reads from a csv file, then process the data a bit with a Python UDF that leverages `sklearn.preprocessing.LabelEncoder`. I have included all necessary files for reproduction in the [GitHub repo](https://github.com/YikSanChan/pyflink-issue-call-already-closed).

To reproduce:
- `conda env create -f environment.yaml`
- `conda activate pyflink-issue-call-already-closed-env`
- `pytest` to verify the udf defined in `ml_udf` works fine
- `python main.py` a few times, and you will see multiple outcomes

There are 3 possible outcomes.

## Outcome 1: success!

It prints 90 expected rows, in a different order from outcome 2 (see below).

## Outcome 2: call already closed

It prints 88 expected rows first, then throws exceptions complaining `java.lang.IllegalStateException: call already closed`.

```
$ python main.py
6> +I(1403227,2,1,5,52,0,25,0,3,2,20,0,0)
7> +I(2278927,5,2,7,236,2,9,1,1347,2,62,0,1)
5> +I(143469,0,2,7,366,2,0,1,1346,2,132,0,1)
1> +I(2689667,5,1,9,329,1,1,0,49,2,86,0,1)
2> +I(3164378,5,2,14,348,2,0,0,1508,2,99,0,0)
5> +I(228014,0,2,0,329,2,0,0,393,2,86,0,1)
1> +I(2722900,5,0,0,200,2,0,0,584,2,63,1,0)
2> +I(3213491,5,1,11,1,2,0,0,656,2,98,0,1)
8> +I(2900644,5,1,7,307,0,1,1,1353,2,138,0,0)
2> +I(3222862,5,2,11,353,0,6,1,1346,2,62,0,1)
5> +I(646044,2,2,4,343,0,14,1,1409,2,48,1,0)
8> +I(2962545,5,2,0,142,2,0,0,501,2,62,1,0)
2> +I(3225216,5,1,8,193,2,0,1,1371,2,96,0,1)
8> +I(3010327,5,1,13,52,2,2,0,26,2,20,0,1)
6> +I(1433504,5,1,0,274,2,0,0,740,2,85,1,0)
8> +I(3013677,5,1,0,56,2,0,0,808,2,82,1,0)
6> +I(1492249,5,2,32,238,2,0,1,1407,2,96,0,1)
7> +I(2357917,5,2,0,365,0,1,0,33,2,54,0,0)
6> +I(1576752,5,2,0,307,2,0,1,1347,2,138,1,0)
8> +I(3015812,5,2,5,335,0,14,0,1287,2,96,0,0)
2> +I(3288417,5,2,6,293,2,13,0,624,2,98,0,1)
6> +I(1588680,5,2,11,144,2,0,1,1346,2,85,0,1)
8> +I(3032974,5,1,0,224,2,0,0,216,2,54,1,0)
2> +I(3289587,5,2,0,296,2,3,0,416,2,54,0,0)
8> +I(3036222,5,2,0,161,2,0,0,1003,2,34,0,0)
5> +I(657365,2,2,0,36,2,0,1,1422,2,62,1,0)
8> +I(3038267,1,1,14,236,2,2,1,1357,2,62,0,1)
1> +I(2729639,5,2,0,380,2,1,0,319,2,129,1,0)
8> +I(3127877,5,0,0,384,2,2,1,1415,2,108,1,0)
2> +I(3306929,5,1,13,232,2,0,0,367,2,54,0,1)
2> +I(3319428,5,2,9,383,0,1,0,481,2,147,0,0)
2> +I(3348282,5,1,0,152,2,0,0,1298,2,82,1,0)
1> +I(2730975,5,2,7,307,2,1,1,1412,2,138,0,1)
6> +I(1663817,5,2,0,193,2,0,0,856,2,96,0,0)
7> +I(2403815,5,1,0,247,2,0,0,567,2,108,1,0)
6> +I(1691686,2,2,0,52,2,0,1,1346,2,20,0,1)
6> +I(1744025,5,2,0,353,2,0,1,1410,2,62,0,0)
1> +I(2757438,5,2,6,346,0,0,0,1124,2,82,0,0)
6> +I(1779238,5,1,32,348,0,0,1,1412,2,99,0,1)
1> +I(2757877,5,1,9,105,2,1,0,1324,2,44,0,1)
4> +I(1951579,5,2,7,250,0,0,0,30,2,62,0,1)
1> +I(2791951,5,2,0,86,2,0,0,812,2,147,0,0)
4> +I(2033542,5,1,0,348,2,0,0,591,2,99,0,1)
2> +I(3404386,5,1,8,375,2,0,1,1409,2,98,0,0)
1> +I(2802070,5,2,0,236,2,0,1,1414,2,62,0,0)
8> +I(3133463,5,2,9,310,2,0,0,68,2,129,0,1)
2> +I(3419962,5,2,0,236,2,2,0,567,2,62,0,0)
1> +I(2824123,5,2,0,365,0,18,1,1354,2,54,1,0)
8> +I(3141633,5,2,13,101,0,22,0,989,2,147,0,0)
5> +I(779727,1,2,10,148,0,1,0,828,2,85,0,0)
1> +I(2863220,5,1,12,383,0,0,0,175,2,147,0,0)
4> +I(2097867,5,1,10,307,0,0,0,399,2,138,0,1)
6> +I(1779859,2,2,0,101,2,1,1,1365,2,147,0,1)
4> +I(2104055,4,2,6,74,2,2,0,1223,2,83,0,1)
6> +I(1918655,4,1,0,304,2,0,0,963,2,98,0,1)
4> +I(2118337,5,2,13,147,2,1,1,1394,2,86,0,1)
4> +I(2176902,5,1,8,215,0,0,0,92,2,132,0,1)
7> +I(2404608,5,2,11,7,2,0,1,1353,2,2,0,1)
4> +I(2207216,5,2,0,161,2,1,1,1421,2,34,0,0)
7> +I(2418491,5,2,11,161,0,1,1,1415,2,34,0,0)
7> +I(2419129,5,1,6,52,0,7,1,1358,2,20,0,0)
4> +I(2218950,5,2,0,14,2,0,0,849,2,107,0,0)
7> +I(2421236,4,2,4,77,0,30,0,596,2,55,0,1)
4> +I(2226603,5,2,6,1,0,1,0,1480,2,108,0,1)
7> +I(2450894,5,2,0,142,0,3,0,579,2,62,0,0)
1> +I(2881859,5,2,11,52,0,1,0,231,2,20,0,0)
4> +I(2272478,5,2,13,238,0,0,0,1288,2,96,0,0)
5> +I(894090,4,2,4,1,2,25,1,1415,1,1,0,0)
4> +I(2276773,5,1,7,88,2,0,0,1166,2,86,0,1)
7> +I(2506290,5,2,8,215,2,0,1,1412,2,132,0,1)
5> +I(962452,5,1,8,259,2,0,0,6,2,62,0,0)
7> +I(2562006,5,1,9,16,2,0,0,1239,2,54,0,1)
5> +I(972543,5,1,7,51,2,0,1,1373,2,14,0,1)
5> +I(1044530,5,2,3,142,2,1,0,231,2,62,0,1)
5> +I(1107922,5,2,12,52,0,0,1,1347,2,20,0,0)
7> +I(2606661,5,2,0,334,2,0,0,1287,2,133,0,0)
5> +I(1128124,2,2,4,1,2,0,1,1418,1,1,0,0)
7> +I(2644346,5,2,9,152,2,0,1,1414,2,82,0,1)
5> +I(1390365,5,0,0,289,2,0,1,1409,2,82,0,0)
3> +I(3426408,5,2,0,278,2,1,0,1121,2,85,1,0)
3> +I(3446903,5,0,0,298,2,0,1,1422,2,132,0,0)
3> +I(3450768,5,2,0,307,2,0,1,1406,2,138,1,0)
3> +I(3463334,5,2,0,365,2,0,0,393,2,54,1,0)
3> +I(3503272,5,2,0,329,2,0,1,1407,2,86,1,0)
3> +I(3505986,5,2,8,52,2,0,1,1409,2,20,1,0)
3> +I(3513234,5,2,4,310,2,1,0,1288,2,129,0,0)
3> +I(3517754,5,0,0,103,2,0,1,1394,2,132,0,1)
3> +I(3575369,5,2,0,270,0,0,1,1415,2,82,1,0)
3> +I(3667690,5,2,2,224,2,0,0,415,2,54,0,1)
3> +I(3676173,5,1,6,230,1,3,1,1347,2,97,0,0)
Apr 16, 2021 11:06:33 AM org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.SerializingExecutor run
SEVERE: Exception while executing runnable org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@6c97b8e5
java.lang.IllegalStateException: closedStatus can only be set once
	at org.apache.beam.vendor.grpc.v1p26p0.com.google.common.base.Preconditions.checkState(Preconditions.java:511)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.AbstractServerStream$TransportState.setClosedStatus(AbstractServerStream.java:351)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.AbstractServerStream$TransportState.access$000(AbstractServerStream.java:188)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.AbstractServerStream.close(AbstractServerStream.java:136)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerCallImpl.closeInternal(ServerCallImpl.java:218)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerCallImpl.close(ServerCallImpl.java:202)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.stub.ServerCalls$ServerCallStreamObserverImpl.onCompleted(ServerCalls.java:371)
	at org.apache.beam.runners.fnexecution.state.GrpcStateService$Inbound.onCompleted(GrpcStateService.java:153)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.stub.ServerCalls$StreamingServerCallHandler$StreamingServerCallListener.onHalfClose(ServerCalls.java:262)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.PartialForwardingServerCallListener.onHalfClose(PartialForwardingServerCallListener.java:35)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.ForwardingServerCallListener.onHalfClose(ForwardingServerCallListener.java:23)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.ForwardingServerCallListener$SimpleForwardingServerCallListener.onHalfClose(ForwardingServerCallListener.java:40)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.Contexts$ContextualizedServerCallListener.onHalfClose(Contexts.java:86)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerCallImpl$ServerStreamListenerImpl.halfClosed(ServerCallImpl.java:331)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed.runInContext(ServerImpl.java:817)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ContextRunnable.run(ContextRunnable.java:37)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.SerializingExecutor.run(SerializingExecutor.java:123)
	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
	at java.lang.Thread.run(Thread.java:748)

Apr 16, 2021 11:06:33 AM org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.SerializingExecutor run
SEVERE: Exception while executing runnable org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@475c446d
java.lang.IllegalStateException: call already closed
	at org.apache.beam.vendor.grpc.v1p26p0.com.google.common.base.Preconditions.checkState(Preconditions.java:511)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerCallImpl.closeInternal(ServerCallImpl.java:209)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerCallImpl.close(ServerCallImpl.java:202)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.stub.ServerCalls$ServerCallStreamObserverImpl.onCompleted(ServerCalls.java:371)
	at org.apache.beam.runners.fnexecution.state.GrpcStateService$Inbound.onCompleted(GrpcStateService.java:153)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.stub.ServerCalls$StreamingServerCallHandler$StreamingServerCallListener.onHalfClose(ServerCalls.java:262)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.PartialForwardingServerCallListener.onHalfClose(PartialForwardingServerCallListener.java:35)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.ForwardingServerCallListener.onHalfClose(ForwardingServerCallListener.java:23)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.ForwardingServerCallListener$SimpleForwardingServerCallListener.onHalfClose(ForwardingServerCallListener.java:40)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.Contexts$ContextualizedServerCallListener.onHalfClose(Contexts.java:86)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerCallImpl$ServerStreamListenerImpl.halfClosed(ServerCallImpl.java:331)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed.runInContext(ServerImpl.java:817)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.ContextRunnable.run(ContextRunnable.java:37)
	at org.apache.beam.vendor.grpc.v1p26p0.io.grpc.internal.SerializingExecutor.run(SerializingExecutor.java:123)
	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
	at java.lang.Thread.run(Thread.java:748)
```

This looks similar to [the thread](https://issues.apache.org/jira/browse/FLINK-17959), but it seems the issue was resolved.

## Outcome 3: NullPointerException

Having noticed the oddities, I append 10 more rows to `users.csv`.

```
3704928,4,2,7,黔东南苗族侗族自治州,1,0,1,iPad Pro 10.5-inch,中国,贵州,1,1
3708233,4,2,100,九江,3,0,1,"iPhone9,1",中国,江西,2,1
3717067,4,1,100,长沙,3,0,1,iPhone 6s Plus,中国,湖南,2,1
3719109,4,1,12,东莞,1,1,0,PEMM00,中国,广东,1,2
3757129,4,2,14,潍坊,3,0,0,SPN-AL00,中国,山东,1,2
3757548,4,0,100,重庆,3,1,0,V1838A,中国,重庆,1,1
3787732,4,1,8,大连,1,0,0,MRX-W09,中国,辽宁,1,2
3816046,4,2,100,UNKNOWN,3,0,1,iPhone 11,中国,湖南,2,1
3824537,4,2,9,株洲,3,1,1,iPad 5,中国,湖南,1,2
3826115,4,2,10,西安,3,0,1,iPhone 8 Plus,中国,陕西,2,1
```

This time the job first prints 88 (AGAIN 88 - the magic number!) expected rows, then throws NullPointerException.

```
$ python main.py
1> +I(3348282,5,1,0,152,2,0,0,1298,2,82,1,0)
5> +I(1433504,5,1,0,274,2,0,0,740,2,85,1,0)
8> +I(2418491,5,2,11,161,0,1,1,1415,2,34,0,0)
5> +I(1492249,5,2,32,238,2,0,1,1407,2,96,0,1)
1> +I(3404386,5,1,8,375,2,0,1,1409,2,98,0,0)
5> +I(1576752,5,2,0,307,2,0,1,1347,2,138,1,0)
8> +I(2419129,5,1,6,52,0,7,1,1358,2,20,0,0)
6> +I(2104055,4,2,6,74,2,2,0,1223,2,83,0,1)
5> +I(1588680,5,2,11,144,2,0,1,1346,2,85,0,1)
8> +I(2421236,4,2,4,77,0,30,0,596,2,55,0,1)
6> +I(2118337,5,2,13,147,2,1,1,1394,2,86,0,1)
7> +I(143469,0,2,7,366,2,0,1,1346,2,132,0,1)
6> +I(2176902,5,1,8,215,0,0,0,92,2,132,0,1)
8> +I(2450894,5,2,0,142,0,3,0,579,2,62,0,0)
3> +I(2757877,5,1,9,105,2,1,0,1324,2,44,0,1)
6> +I(2207216,5,2,0,161,2,1,1,1421,2,34,0,0)
8> +I(2506290,5,2,8,215,2,0,1,1412,2,132,0,1)
3> +I(2791951,5,2,0,86,2,0,0,812,2,147,0,0)
6> +I(2218950,5,2,0,14,2,0,0,849,2,107,0,0)
4> +I(3036222,5,2,0,161,2,0,0,1003,2,34,0,0)
8> +I(2562006,5,1,9,16,2,0,0,1239,2,54,0,1)
6> +I(2226603,5,2,6,1,0,1,0,1480,2,108,0,1)
5> +I(1663817,5,2,0,193,2,0,0,856,2,96,0,0)
6> +I(2272478,5,2,13,238,0,0,0,1288,2,96,0,0)
4> +I(3038267,1,1,14,236,2,2,1,1357,2,62,0,1)
5> +I(1691686,2,2,0,52,2,0,1,1346,2,20,0,1)
8> +I(2606661,5,2,0,334,2,0,0,1287,2,133,0,0)
4> +I(3127877,5,0,0,384,2,2,1,1415,2,108,1,0)
8> +I(2644346,5,2,9,152,2,0,1,1414,2,82,0,1)
4> +I(3133463,5,2,9,310,2,0,0,68,2,129,0,1)
7> +I(228014,0,2,0,329,2,0,0,393,2,86,0,1)
8> +I(2689667,5,1,9,329,1,1,0,49,2,86,0,1)
4> +I(3141633,5,2,13,101,0,22,0,989,2,147,0,0)
7> +I(646044,2,2,4,343,0,14,1,1409,2,48,1,0)
8> +I(2722900,5,0,0,200,2,0,0,584,2,63,1,0)
4> +I(3164378,5,2,14,348,2,0,0,1508,2,99,0,0)
8> +I(2729639,5,2,0,380,2,1,0,319,2,129,1,0)
4> +I(3213491,5,1,11,1,2,0,0,656,2,98,0,1)
8> +I(2730975,5,2,7,307,2,1,1,1412,2,138,0,1)
4> +I(3222862,5,2,11,353,0,6,1,1346,2,62,0,1)
8> +I(2757438,5,2,6,346,0,0,0,1124,2,82,0,0)
1> +I(3419962,5,2,0,236,2,2,0,567,2,62,0,0)
5> +I(1744025,5,2,0,353,2,0,1,1410,2,62,0,0)
1> +I(3426408,5,2,0,278,2,1,0,1121,2,85,1,0)
7> +I(657365,2,2,0,36,2,0,1,1422,2,62,1,0)
7> +I(779727,1,2,10,148,0,1,0,828,2,85,0,0)
4> +I(3225216,5,1,8,193,2,0,1,1371,2,96,0,1)
6> +I(2276773,5,1,7,88,2,0,0,1166,2,86,0,1)
4> +I(3288417,5,2,6,293,2,13,0,624,2,98,0,1)
3> +I(2802070,5,2,0,236,2,0,1,1414,2,62,0,0)
4> +I(3289587,5,2,0,296,2,3,0,416,2,54,0,0)
3> +I(2824123,5,2,0,365,0,18,1,1354,2,54,1,0)
6> +I(2278927,5,2,7,236,2,9,1,1347,2,62,0,1)
3> +I(2863220,5,1,12,383,0,0,0,175,2,147,0,0)
7> +I(894090,4,2,4,1,2,25,1,1415,1,1,0,0)
3> +I(2881859,5,2,11,52,0,1,0,231,2,20,0,0)
7> +I(962452,5,1,8,259,2,0,0,6,2,62,0,0)
3> +I(2900644,5,1,7,307,0,1,1,1353,2,138,0,0)
7> +I(972543,5,1,7,51,2,0,1,1373,2,14,0,1)
1> +I(3446903,5,0,0,298,2,0,1,1422,2,132,0,0)
1> +I(3450768,5,2,0,307,2,0,1,1406,2,138,1,0)
5> +I(1779238,5,1,32,348,0,0,1,1412,2,99,0,1)
7> +I(1044530,5,2,3,142,2,1,0,231,2,62,0,1)
5> +I(1779859,2,2,0,101,2,1,1,1365,2,147,0,1)
1> +I(3463334,5,2,0,365,2,0,0,393,2,54,1,0)
5> +I(1918655,4,1,0,304,2,0,0,963,2,98,0,1)
1> +I(3503272,5,2,0,329,2,0,1,1407,2,86,1,0)
3> +I(2962545,5,2,0,142,2,0,0,501,2,62,1,0)
1> +I(3505986,5,2,8,52,2,0,1,1409,2,20,1,0)
3> +I(3010327,5,1,13,52,2,2,0,26,2,20,0,1)
6> +I(2357917,5,2,0,365,0,1,0,33,2,54,0,0)
3> +I(3013677,5,1,0,56,2,0,0,808,2,82,1,0)
6> +I(2403815,5,1,0,247,2,0,0,567,2,108,1,0)
4> +I(3306929,5,1,13,232,2,0,0,367,2,54,0,1)
6> +I(2404608,5,2,11,7,2,0,1,1353,2,2,0,1)
4> +I(3319428,5,2,9,383,0,1,0,481,2,147,0,0)
3> +I(3015812,5,2,5,335,0,14,0,1287,2,96,0,0)
1> +I(3513234,5,2,4,310,2,1,0,1288,2,129,0,0)
3> +I(3032974,5,1,0,224,2,0,0,216,2,54,1,0)
1> +I(3517754,5,0,0,103,2,0,1,1394,2,132,0,1)
1> +I(3575369,5,2,0,270,0,0,1,1415,2,82,1,0)
5> +I(1951579,5,2,7,250,0,0,0,30,2,62,0,1)
5> +I(2033542,5,1,0,348,2,0,0,591,2,99,0,1)
7> +I(1107922,5,2,12,52,0,0,1,1347,2,20,0,0)
5> +I(2097867,5,1,10,307,0,0,0,399,2,138,0,1)
7> +I(1128124,2,2,4,1,2,0,1,1418,1,1,0,0)
7> +I(1390365,5,0,0,289,2,0,1,1409,2,82,0,0)
7> +I(1403227,2,1,5,52,0,25,0,3,2,20,0,0)
Traceback (most recent call last):
  File "main.py", line 109, in <module>
    t_env.execute_sql(TRANSFORM_DML).wait()
  File "/usr/local/anaconda3/envs/featflow-ml-env/lib/python3.7/site-packages/pyflink/table/table_result.py", line 76, in wait
    get_method(self._j_table_result, "await")()
  File "/usr/local/anaconda3/envs/featflow-ml-env/lib/python3.7/site-packages/py4j/java_gateway.py", line 1286, in __call__
    answer, self.gateway_client, self.target_id, self.name)
  File "/usr/local/anaconda3/envs/featflow-ml-env/lib/python3.7/site-packages/pyflink/util/exceptions.py", line 147, in deco
    return f(*a, **kw)
  File "/usr/local/anaconda3/envs/featflow-ml-env/lib/python3.7/site-packages/py4j/protocol.py", line 328, in get_return_value
    format(target_id, ".", name), value)
py4j.protocol.Py4JJavaError: An error occurred while calling o51.await.
: java.util.concurrent.ExecutionException: org.apache.flink.table.api.TableException: Failed to wait job finish
	at java.util.concurrent.CompletableFuture.reportGet(CompletableFuture.java:357)
	at java.util.concurrent.CompletableFuture.get(CompletableFuture.java:1908)
	at org.apache.flink.table.api.internal.TableResultImpl.awaitInternal(TableResultImpl.java:119)
	at org.apache.flink.table.api.internal.TableResultImpl.await(TableResultImpl.java:86)
	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.lang.reflect.Method.invoke(Method.java:498)
	at org.apache.flink.api.python.shaded.py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
	at org.apache.flink.api.python.shaded.py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:357)
	at org.apache.flink.api.python.shaded.py4j.Gateway.invoke(Gateway.java:282)
	at org.apache.flink.api.python.shaded.py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
	at org.apache.flink.api.python.shaded.py4j.commands.CallCommand.execute(CallCommand.java:79)
	at org.apache.flink.api.python.shaded.py4j.GatewayConnection.run(GatewayConnection.java:238)
	at java.lang.Thread.run(Thread.java:748)
Caused by: org.apache.flink.table.api.TableException: Failed to wait job finish
	at org.apache.flink.table.api.internal.InsertResultIterator.hasNext(InsertResultIterator.java:59)
	at org.apache.flink.table.api.internal.TableResultImpl$CloseableRowIteratorWrapper.hasNext(TableResultImpl.java:355)
	at org.apache.flink.table.api.internal.TableResultImpl$CloseableRowIteratorWrapper.isFirstRowReady(TableResultImpl.java:368)
	at org.apache.flink.table.api.internal.TableResultImpl.lambda$awaitInternal$1(TableResultImpl.java:107)
	at java.util.concurrent.CompletableFuture$AsyncRun.run(CompletableFuture.java:1640)
	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
	... 1 more
Caused by: java.util.concurrent.ExecutionException: org.apache.flink.runtime.client.JobExecutionException: Job execution failed.
	at java.util.concurrent.CompletableFuture.reportGet(CompletableFuture.java:357)
	at java.util.concurrent.CompletableFuture.get(CompletableFuture.java:1908)
	at org.apache.flink.table.api.internal.InsertResultIterator.hasNext(InsertResultIterator.java:57)
	... 7 more
Caused by: org.apache.flink.runtime.client.JobExecutionException: Job execution failed.
	at org.apache.flink.runtime.jobmaster.JobResult.toJobExecutionResult(JobResult.java:147)
	at org.apache.flink.runtime.minicluster.MiniClusterJobClient.lambda$getJobExecutionResult$2(MiniClusterJobClient.java:119)
	at java.util.concurrent.CompletableFuture.uniApply(CompletableFuture.java:616)
	at java.util.concurrent.CompletableFuture$UniApply.tryFire(CompletableFuture.java:591)
	at java.util.concurrent.CompletableFuture.postComplete(CompletableFuture.java:488)
	at java.util.concurrent.CompletableFuture.complete(CompletableFuture.java:1975)
	at org.apache.flink.runtime.rpc.akka.AkkaInvocationHandler.lambda$invokeRpc$0(AkkaInvocationHandler.java:229)
	at java.util.concurrent.CompletableFuture.uniWhenComplete(CompletableFuture.java:774)
	at java.util.concurrent.CompletableFuture$UniWhenComplete.tryFire(CompletableFuture.java:750)
	at java.util.concurrent.CompletableFuture.postComplete(CompletableFuture.java:488)
	at java.util.concurrent.CompletableFuture.complete(CompletableFuture.java:1975)
	at org.apache.flink.runtime.concurrent.FutureUtils$1.onComplete(FutureUtils.java:996)
	at akka.dispatch.OnComplete.internal(Future.scala:264)
	at akka.dispatch.OnComplete.internal(Future.scala:261)
	at akka.dispatch.japi$CallbackBridge.apply(Future.scala:191)
	at akka.dispatch.japi$CallbackBridge.apply(Future.scala:188)
	at scala.concurrent.impl.CallbackRunnable.run(Promise.scala:36)
	at org.apache.flink.runtime.concurrent.Executors$DirectExecutionContext.execute(Executors.java:74)
	at scala.concurrent.impl.CallbackRunnable.executeWithValue(Promise.scala:44)
	at scala.concurrent.impl.Promise$DefaultPromise.tryComplete(Promise.scala:252)
	at akka.pattern.PromiseActorRef.$bang(AskSupport.scala:572)
	at akka.pattern.PipeToSupport$PipeableFuture$$anonfun$pipeTo$1.applyOrElse(PipeToSupport.scala:22)
	at akka.pattern.PipeToSupport$PipeableFuture$$anonfun$pipeTo$1.applyOrElse(PipeToSupport.scala:21)
	at scala.concurrent.Future$$anonfun$andThen$1.apply(Future.scala:436)
	at scala.concurrent.Future$$anonfun$andThen$1.apply(Future.scala:435)
	at scala.concurrent.impl.CallbackRunnable.run(Promise.scala:36)
	at akka.dispatch.BatchingExecutor$AbstractBatch.processBatch(BatchingExecutor.scala:55)
	at akka.dispatch.BatchingExecutor$BlockableBatch$$anonfun$run$1.apply$mcV$sp(BatchingExecutor.scala:91)
	at akka.dispatch.BatchingExecutor$BlockableBatch$$anonfun$run$1.apply(BatchingExecutor.scala:91)
	at akka.dispatch.BatchingExecutor$BlockableBatch$$anonfun$run$1.apply(BatchingExecutor.scala:91)
	at scala.concurrent.BlockContext$.withBlockContext(BlockContext.scala:72)
	at akka.dispatch.BatchingExecutor$BlockableBatch.run(BatchingExecutor.scala:90)
	at akka.dispatch.TaskInvocation.run(AbstractDispatcher.scala:40)
	at akka.dispatch.ForkJoinExecutorConfigurator$AkkaForkJoinTask.exec(ForkJoinExecutorConfigurator.scala:44)
	at akka.dispatch.forkjoin.ForkJoinTask.doExec(ForkJoinTask.java:260)
	at akka.dispatch.forkjoin.ForkJoinPool$WorkQueue.runTask(ForkJoinPool.java:1339)
	at akka.dispatch.forkjoin.ForkJoinPool.runWorker(ForkJoinPool.java:1979)
	at akka.dispatch.forkjoin.ForkJoinWorkerThread.run(ForkJoinWorkerThread.java:107)
Caused by: org.apache.flink.runtime.JobException: Recovery is suppressed by NoRestartBackoffTimeStrategy
	at org.apache.flink.runtime.executiongraph.failover.flip1.ExecutionFailureHandler.handleFailure(ExecutionFailureHandler.java:116)
	at org.apache.flink.runtime.executiongraph.failover.flip1.ExecutionFailureHandler.getFailureHandlingResult(ExecutionFailureHandler.java:78)
	at org.apache.flink.runtime.scheduler.DefaultScheduler.handleTaskFailure(DefaultScheduler.java:224)
	at org.apache.flink.runtime.scheduler.DefaultScheduler.maybeHandleTaskFailure(DefaultScheduler.java:217)
	at org.apache.flink.runtime.scheduler.DefaultScheduler.updateTaskExecutionStateInternal(DefaultScheduler.java:208)
	at org.apache.flink.runtime.scheduler.SchedulerBase.updateTaskExecutionState(SchedulerBase.java:610)
	at org.apache.flink.runtime.scheduler.SchedulerNG.updateTaskExecutionState(SchedulerNG.java:89)
	at org.apache.flink.runtime.jobmaster.JobMaster.updateTaskExecutionState(JobMaster.java:419)
	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.lang.reflect.Method.invoke(Method.java:498)
	at org.apache.flink.runtime.rpc.akka.AkkaRpcActor.handleRpcInvocation(AkkaRpcActor.java:286)
	at org.apache.flink.runtime.rpc.akka.AkkaRpcActor.handleRpcMessage(AkkaRpcActor.java:201)
	at org.apache.flink.runtime.rpc.akka.FencedAkkaRpcActor.handleRpcMessage(FencedAkkaRpcActor.java:74)
	at org.apache.flink.runtime.rpc.akka.AkkaRpcActor.handleMessage(AkkaRpcActor.java:154)
	at akka.japi.pf.UnitCaseStatement.apply(CaseStatements.scala:26)
	at akka.japi.pf.UnitCaseStatement.apply(CaseStatements.scala:21)
	at scala.PartialFunction$class.applyOrElse(PartialFunction.scala:123)
	at akka.japi.pf.UnitCaseStatement.applyOrElse(CaseStatements.scala:21)
	at scala.PartialFunction$OrElse.applyOrElse(PartialFunction.scala:170)
	at scala.PartialFunction$OrElse.applyOrElse(PartialFunction.scala:171)
	at scala.PartialFunction$OrElse.applyOrElse(PartialFunction.scala:171)
	at akka.actor.Actor$class.aroundReceive(Actor.scala:517)
	at akka.actor.AbstractActor.aroundReceive(AbstractActor.scala:225)
	at akka.actor.ActorCell.receiveMessage(ActorCell.scala:592)
	at akka.actor.ActorCell.invoke(ActorCell.scala:561)
	at akka.dispatch.Mailbox.processMailbox(Mailbox.scala:258)
	at akka.dispatch.Mailbox.run(Mailbox.scala:225)
	at akka.dispatch.Mailbox.exec(Mailbox.scala:235)
	... 4 more
Caused by: org.apache.flink.streaming.runtime.tasks.AsynchronousException: Caught exception while processing timer.
	at org.apache.flink.streaming.runtime.tasks.StreamTask$StreamTaskAsyncExceptionHandler.handleAsyncException(StreamTask.java:1108)
	at org.apache.flink.streaming.runtime.tasks.StreamTask.handleAsyncException(StreamTask.java:1082)
	at org.apache.flink.streaming.runtime.tasks.StreamTask.invokeProcessingTimeCallback(StreamTask.java:1213)
	at org.apache.flink.streaming.runtime.tasks.StreamTask.lambda$null$17(StreamTask.java:1202)
	at org.apache.flink.streaming.runtime.tasks.StreamTaskActionExecutor$SynchronizedStreamTaskActionExecutor.runThrowing(StreamTaskActionExecutor.java:92)
	at org.apache.flink.streaming.runtime.tasks.mailbox.Mail.run(Mail.java:78)
	at org.apache.flink.streaming.runtime.tasks.mailbox.MailboxExecutorImpl.tryYield(MailboxExecutorImpl.java:91)
	at org.apache.flink.streaming.runtime.tasks.StreamOperatorWrapper.quiesceTimeServiceAndCloseOperator(StreamOperatorWrapper.java:155)
	at org.apache.flink.streaming.runtime.tasks.StreamOperatorWrapper.close(StreamOperatorWrapper.java:130)
	at org.apache.flink.streaming.runtime.tasks.OperatorChain.closeOperators(OperatorChain.java:412)
	at org.apache.flink.streaming.runtime.tasks.StreamTask.afterInvoke(StreamTask.java:585)
	at org.apache.flink.streaming.runtime.tasks.StreamTask.invoke(StreamTask.java:547)
	at org.apache.flink.runtime.taskmanager.Task.doRun(Task.java:722)
	at org.apache.flink.runtime.taskmanager.Task.run(Task.java:547)
	at java.lang.Thread.run(Thread.java:748)
Caused by: TimerException{java.lang.RuntimeException: Failed to close remote bundle}
	... 13 more
Caused by: java.lang.RuntimeException: Failed to close remote bundle
	at org.apache.flink.streaming.api.runners.python.beam.BeamPythonFunctionRunner.finishBundle(BeamPythonFunctionRunner.java:371)
	at org.apache.flink.streaming.api.runners.python.beam.BeamPythonFunctionRunner.flush(BeamPythonFunctionRunner.java:325)
	at org.apache.flink.streaming.api.operators.python.AbstractPythonFunctionOperator.invokeFinishBundle(AbstractPythonFunctionOperator.java:291)
	at org.apache.flink.table.runtime.operators.python.scalar.arrow.RowDataArrowPythonScalarFunctionOperator.invokeFinishBundle(RowDataArrowPythonScalarFunctionOperator.java:77)
	at org.apache.flink.streaming.api.operators.python.AbstractPythonFunctionOperator.checkInvokeFinishBundleByTime(AbstractPythonFunctionOperator.java:285)
	at org.apache.flink.streaming.api.operators.python.AbstractPythonFunctionOperator.lambda$open$0(AbstractPythonFunctionOperator.java:134)
	at org.apache.flink.streaming.runtime.tasks.StreamTask.invokeProcessingTimeCallback(StreamTask.java:1211)
	... 12 more
Caused by: java.lang.NullPointerException
	at org.apache.flink.streaming.api.runners.python.beam.BeamPythonFunctionRunner.finishBundle(BeamPythonFunctionRunner.java:369)
	... 18 more
```

The NullPointerException reminds me of [this question](https://stackoverflow.com/questions/67092978/pyflink-vectorized-udf-throws-nullpointerexception), but I have passed the `test_ml_udf.py` to ensure both the input and output types are `pandas.Series` with same length.

## Why?
