# class-code


### Extra class
If you understand the code, try this
Mini LLM Serving Syste


The entire assignment is about one idea: The GPU is scarce. Decide carefully what work enters, what work runs next, and where it runs. Then measure whether your decisions actually helped.



For your final project, you will build this with real models and on actual GPUs



You are going to build a fake LLM serving system. There is no real model and no GPU. Imagine 100 users sending requests to an LLM server, but your server has limited capacity. Your system has to answer three questions:

Should I accept this request? → Admission control

Which request should run next? → Scheduler

Which worker/GPU should receive it? → Router

Finally, you will connect all three and test the complete system.



The request
Every request has:


JavaScript
id
arrival_t
priority          # 0 = important/interactive, 20 = batch
prompt_tokens     # size of the input
max_new_tokens    # maximum output size
prefix_hash       # identifies shared input, or None
timeout_s         # how long the user is willing to wait
tenant            # customer/user group

Your fake server has limited:


JavaScript
decode_slots      # how many requests can generate tokens at once
KV blocks         # memory available for running requests
prefill tokens    # how much input we can process per step

Think of these as the server's three scarce resources.



This exercise has 4 parts.



Part 1. Admission Control - Should we accept the request?
File: admit.py

Write:


Python
should_shed(req, snap)

It answers:

"Is the server too busy to safely accept this request?"

Return:


JavaScript
(shed?, code, retry_after_seconds)



First: write these tests


Your tests must show:




JavaScript
| Situation | Result |
| :--- | :--- |
| Tenant has used 96% of token allowance | reject with 429 |
| Tenant has used 96% of request allowance | reject with 429 |
| Request would probably wait > half its timeout | reject with 503/529 |
| Only 5% KV memory remains + new prefix | reject with 503/529 |
| Only 5% KV memory remains + prefix already exists | accept |
| Very bad tail latency + interactive request | accept |
| Very bad tail latency + batch request | reject |


Then implement these rules


1. Protect tenants.

Check the tenant's token limit before its request-count limit.

Why? Ten requests are not necessarily ten times the same amount of work.



2. Don't accept requests that are already doomed.

Estimate:


JavaScript
expected queue wait = queue length × p50 TTFT

If that is more than half of the request's timeout, reject it.



3. Don't run out of KV memory.

If less than 8% of KV memory remains, reject a request whose prefix is not already cached. A request using a prefix we already have is allowed in.



4. Protect interactive users.

If p99 latency is more than 4× p50 latency and the queue is growing:

keep priority < 10 requests;

reject priority >= 10 requests.

Important: admission control does not retry requests or move them to another worker. It only says accept or reject.



Use:


JavaScript
429 = tenant limit
503/529 = server capacity

Short question
In about ½ page, explain which of your rules represents the ideas behind:

DALL·E's 5-minute cancellation,

Anthropic's late-capacity behavior,

Cloudflare overload protection.



Part 2. Scheduler - Which request gets the GPU?
File: sched.py



Now pretend a request has been accepted. And your fake GPU can only do a limited amount of work per step.



Write:


Python
step(waiting, running, budget)

Each call to step() means: "The GPU gets one more chance to do work."



You must support three ways of choosing requests:


JavaScript
fcfs
priority
drr

So:


JavaScript
priority 0 → before priority 20

If priorities are equal, use arrival time.



Rules to add on scheduler:

Don't process huge prompts all at once: If a request has a 32,000-token prompt but the step budget is 2,048, do not process all 32,000 tokens at once. Process at most 2,048 this step and continue later. This is chunked prefill.

Don't let one huge prompt block decodingAfter processing at most one prefill chunk, use the remaining capacity for requests that are already decoding. In other words:


JavaScript
one prefill chunk
        ↓
use whatever remains for decoding



If KV memory runs out, preempt i.e. stop the lowest-priority running request. Throw away its KV memory and put it back into the waiting queue. When it runs again, it has to recompute its prompt. This is intentionally preempt, don't swap.

Count:


JavaScript
preempts
wasted_decode_tokens


If the client disconnects,


Python
req.aborted == True

then, remove it immediately. Free its KV memory and do not generate any more tokens for it.



Count:


JavaScript
aborted_freed





Now, run this workload
Create:


JavaScript
traces/mixed.jsonl

containing:


JavaScript
70% interactive
  priority 0
  prompt: 200–800
  output: 64–256

20% batch
  priority 20
  prompt: 2,000–8,000
  output: 512–2,048

10% shared-prefix agents
  same prefix
  prompt: 4,000
  3,500 tokens are shared

Run for 60 simulated seconds using:


JavaScript
FCFS
Priority
DRR

Measure:


JavaScript
completed requests / second
interactive p99 TTFT
batch p99 TTFT
preempts / second
wasted decode tokens
requests rejected

Answer:
Which scheduler wastes the most decode work? Explain why in one paragraph.



Part 3. Router - Which worker/GPU should get the request?


File: router.py



Now create two fake workers:


JavaScript
Worker A
Worker B

Each worker tells you:


JavaScript
how much KV memory is free
how many requests are running
how many are waiting
which prefixes it has cached
its p99 latency
whether it is healthy

Write:


Python
pick(req, workers)

It chooses where the request goes.

Support:


JavaScript
random
least_loaded
p2c
prefix_then_load



Add 3 safety rules to your router:

Missing information does NOT mean zero load: If a worker is unhealthy or its load score is unknown:


JavaScript
unknown ≠ idle

Only use unknown workers if all workers are unknown. This is H6.



Don't bounce an overloaded request foreverBefore choosing a worker, check whether each worker would reject the request using your L1 admission logic.



If both workers reject it:


JavaScript
return Shed(503, retry_after=2)

Do not go A → B → A → B.

This is H4.



Router experiments:


Run three traces.

T1 - No shared prefixes
Every request has a different prefix.

Question: Does P2C balance load better than random?

T2 - Lots of shared prefixes
40% of requests share one prefix.

Question: Does prefix-aware routing save KV memory?

You should see approximately:


JavaScript
KV(prefix_then_load)
    <
0.4 × KV(least_loaded)

T3 - Bad/stale information
Make worker B's information 15 seconds old.

Its cached information says:


JavaScript
"B is empty"

even though B is actually busy.

Show that least_loaded can send too much traffic to B.

Then show that your unknown/stale-telemetry handling prevents this mistake.

Report:


JavaScript
p99 TTFT
KV allocated
shed %
traffic sent to stale B

for all four routing strategies.



Part 4. Put everything together
File: serve.py

Your complete system should behave like:


JavaScript
client
   ↓
Should we accept it?
   ↓
admit()
   ↓
Which worker?
   ↓
pick()
   ↓
worker queue
   ↓
GPU step
   ↓
step()

Use:

one process;

two fake workers;

one simulated clock.



Submit

JavaScript
admit.py
sched.py
router.py
serve.py

tests/ (only if you are doing advanced)
traces/
plots/soak.png

REPORT.pdf       # ≤ 2 pages (all tables with short explanations)



By the end of this, you should be able to answer these
Where do I prevent accepting work that will time out?

Where do I protect KV memory?

Where do I prioritize interactive traffic?

Where do I prevent one tenant from monopolizing the GPU?

Where do I preempt a request?

Where do I exploit shared prefixes?

Where do I handle missing worker telemetry?

Where do I stop failover from making overload worse?

=



Advanced:


These tests basically test you on real postmortems within OpenAI, Anthropic and if you can fix the problems like their engineering teams do



1. DALL·E soak
Set:


JavaScript
timeout = 5 seconds
p50 job time = 2 seconds

Keep sending requests until estimated queue wait exceeds 2.5 seconds. Your system must start shedding before completed requests fall to zero.



Create:


JavaScript
plots/soak.png

showing:


JavaScript
admitted
completed
shed

over time.



2. Worker failure and recovery
Kill worker B.

→ Admission should decrease.

Bring B back.

→ Gradually increase admission from 10% to 100% over 30 seconds.

Only continue increasing while:


JavaScript
p99 TTFT < 2 × baseline

If latency exceeds that:

→ stop increasing and shed traffic.



3. Correct error code
Tenant over token limit + empty fleet:


JavaScript
429

NOT 503.

Fleet out of KV + tenant under its limit:


JavaScript
503/529

NOT 429.



4. Prefix stickiness
On T2:


JavaScript
prefix_then_load KV
    <
0.4 × least_loaded KV



5. Abort
Abort a request during decoding.

The next step() must:

free its KV;

increment aborted_freed;

generate zero additional tokens.



