// Render with the IRSA role ARN and image tag from CI, e.g.:
//   jsonnet --ext-str image=ghcr.io/you/finagent:$GIT_SHA \
//           --ext-str role_arn=$(terraform -chdir=infra/terraform output -raw app_irsa_role_arn) \
//           -m deploy/k8s/ infra/jsonnet/environments/prod.jsonnet

local finagent = import '../lib/finagent.libsonnet';

local namespace = 'finagent';
local image = std.extVar('image');
local roleArn = std.extVar('role_arn');
local envFrom = [{ secretRef: { name: 'finagent-secrets' } }];

{
  // Numeric prefixes control apply order for `kubectl apply -f deploy/k8s/`,
  // which applies files in alphabetical order — the namespace must exist
  // before anything that lives in it.
  '00-namespace.json': finagent.namespace(namespace),
  '10-serviceaccount.json': finagent.serviceAccount('finagent', namespace, roleArn),
  '20-deployment.json': finagent.deployment('finagent', image, namespace, replicas=2, envFrom=envFrom),
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
