// Fill out your copyright notice in the Description page of Project Settings.


#include "EndEffector.h"

// Sets default values
AEndEffector::AEndEffector()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

    // Create and initialize the Physics Mesh
    PhysicsMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PhysicsMesh"));
    SetRootComponent(PhysicsMesh);
    PhysicsMesh->SetSimulatePhysics(true);
    PhysicsMesh->SetMassOverrideInKg(NAME_None, mass); // Set mass of the actor

    // Create and initialize the Cable Pin
    CablePin = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("CablePin"));
    CablePin->SetupAttachment(PhysicsMesh);

}

// Called when the game starts or when spawned
void AEndEffector::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AEndEffector::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

