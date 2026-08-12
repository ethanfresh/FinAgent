local finagent = import '../lib/finagent.libsonnet';

local namespace = 'finagent';
local image = 'finagent:local';
local envFrom = [{ secretRef: { name: 'finagent-secrets' } }];

{
  // Numeric prefixes control apply order for `kubectl apply -f deploy/k8s/`,
  // which applies files in alphabetical order — the namespace must exist
  // before anything that lives in it.
  '00-namespace.json': finagent.namespace(namespace),
  '10-serviceaccount.json': finagent.serviceAccount('finagent', namespace, 'arn:aws:iam::000000000000:role/finagent-app-role'),
  '20-deployment.json': finagent.deployment('finagent', image, namespace, replicas=1, envFrom=envFrom),
  '20-service.json': finagent.service('finagent', namespace),
  '30-canary-cronjob.json': finagent.cronJob(
    'finagent-canary',
    '0 6 * * *',
    image,
    ['uv', 'run', 'finagent', 'canary', '--threshold', '0.85'],
    namespace,
    envFrom=envFrom,
  ),
}
