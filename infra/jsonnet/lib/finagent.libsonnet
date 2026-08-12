{
  deployment(name, image, namespace, replicas=2, envFrom=[], serviceAccountName='finagent'):: {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name: name, namespace: namespace, labels: { app: name } },
    spec: {
      replicas: replicas,
      selector: { matchLabels: { app: name } },
      template: {
        metadata: { labels: { app: name } },
        spec: {
          serviceAccountName: serviceAccountName,
          containers: [
            {
              name: name,
              image: image,
              ports: [{ containerPort: 8000 }],
              envFrom: envFrom,
              readinessProbe: {
                httpGet: { path: '/metrics', port: 8000 },
                initialDelaySeconds: 5,
                periodSeconds: 10,
              },
              livenessProbe: {
                httpGet: { path: '/metrics', port: 8000 },
                initialDelaySeconds: 15,
                periodSeconds: 20,
              },
              resources: {
                requests: { cpu: '250m', memory: '512Mi' },
                limits: { cpu: '1', memory: '1Gi' },
              },
            },
          ],
        },
      },
    },
  },

  service(name, namespace, port=80, targetPort=8000):: {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name: name, namespace: namespace },
    spec: {
      selector: { app: name },
      ports: [{ port: port, targetPort: targetPort }],
    },
  },

  serviceAccount(name, namespace, roleArn):: {
    apiVersion: 'v1',
    kind: 'ServiceAccount',
    metadata: {
      name: name,
      namespace: namespace,
      annotations: { 'eks.amazonaws.com/role-arn': roleArn },
    },
  },

  namespace(name):: {
    apiVersion: 'v1',
    kind: 'Namespace',
    metadata: { name: name },
  },

  cronJob(name, schedule, image, command, namespace, envFrom=[], serviceAccountName='finagent'):: {
    apiVersion: 'batch/v1',
    kind: 'CronJob',
    metadata: { name: name, namespace: namespace },
    spec: {
      schedule: schedule,
      jobTemplate: {
        spec: {
          template: {
            spec: {
              serviceAccountName: serviceAccountName,
              restartPolicy: 'OnFailure',
              containers: [
                {
                  name: name,
                  image: image,
                  command: command,
                  envFrom: envFrom,
                },
              ],
            },
          },
        },
      },
    },
  },
}
