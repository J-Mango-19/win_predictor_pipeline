### Checking logs setup.sh run on EC2 instance

1. Get command ID
```
aws ssm send-command \
  --instance-ids i-xxxxxxxxxxxx \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cat /var/log/cloud-init-output.log"]'
```

2. Use command ID to get logs
```
aws ssm get-command-invocation \
  --command-id <command-id-from-above> \
  --instance-id i-xxxxxxxxxxxx
```


### Console output
I haven't seen this do anything yet, but... 
`aws ec2 get-console-output --instance-id i-xxxxxxxxxxxx --output text`