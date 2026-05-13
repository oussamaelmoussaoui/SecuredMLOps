{{/*
Nom complet de l'app
*/}}
{{- define "ids-api.fullname" -}}
{{- printf "%s" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Labels communs
*/}}
{{- define "ids-api.labels" -}}
app: ids-api
app.kubernetes.io/name: ids-api
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: securedmlops
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "ids-api.selectorLabels" -}}
app: ids-api
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
