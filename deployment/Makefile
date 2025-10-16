.PHONY: build-and-push-docker-image build-docker-image tag-docker-image push-docker-image things

# Either python, julia, or r
PROJECT := $(or $(PROJECT), "dev")
# Set default TAG if not passed
TAG := $(or $(TAG),$(shell date +%Y%m%d%H%M))
ROOT_DIR=$(realpath ..)

build-docker-image:
	DOCKER_DEFAULT_PLATFORM="linux/amd64" docker build -f $(ROOT_DIR)/deployment/Dockerfile_$(PROJECT) -t eopf-toolkit-$(PROJECT) $(ROOT_DIR)

tag-docker-image:
	docker tag eopf-toolkit-$(PROJECT) 4zm3809f.c1.de1.container-registry.ovh.net/eopf-toolkit-$(PROJECT)/eopf-toolkit-$(PROJECT):$(TAG)

push-docker-image:
	docker push 4zm3809f.c1.de1.container-registry.ovh.net/eopf-toolkit-$(PROJECT)/eopf-toolkit-$(PROJECT):$(TAG)

build-and-push-docker-image: build-docker-image tag-docker-image push-docker-image
	@echo "Docker image built and pushed with tag: $(TAG)"
