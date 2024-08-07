// Fill out your copyright notice in the Description page of Project Settings.


#include "Environment.h"
#include "GameFramework/Actor.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/World.h"

// MyGameModeBase.cpp

AEnvironment::AEnvironment()
{
    // Set this game mode to call Tick() every frame
    PrimaryActorTick.bCanEverTick = true;

    Reward = 0.0f;
    MaxDeviation = 50.0f; // 설정된 단위에 따라 조정

    // 경로 설정
    TargetPositions = {
        FVector(0.0f, 0.0f, 0.0f),
        FVector(-112.5f, 75.0f, 112.5f),
        FVector(-225.0f, 150.0f, 225.0f),
        FVector(-337.5f, 225.0f, 337.5f),
        FVector(-450.0f, 300.0f, 450.0f)
    };
}

void AEnvironment::BeginPlay()
{
    Super::BeginPlay();
    ResetEnvironment();
}

void AEnvironment::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    // Update environment state
}

void AEnvironment::RestartCurrentLevel()
{
    // 현재 월드 가져오기
    UWorld* World = GetWorld();
    if (World == nullptr)
    {
        return;
    }

    // 현재 레벨 이름 가져오기
    FString CurrentLevelName = World->GetMapName();
    CurrentLevelName.RemoveFromStart(World->StreamingLevelsPrefix);

    // 현재 레벨 다시 불러오기
    FName CurrentLevelFName(*CurrentLevelName);
    UGameplayStatics::OpenLevel(this, CurrentLevelFName);
}


void AEnvironment::ResetEnvironment() {
    

    // 보상을 0으로
    done = false;
    Reward = 0.0f;

    // 엔드이펙터 생성
    // 월드를 가져옵니다.
    UWorld* World = GetWorld();
    if (World && BP_EndEffector)
    {
        AActor* SpawnedActor2 = World->SpawnActor<AActor>(BP_EndEffector, FVector(0.0f, 0.0f, 225.0f), FRotator(0.0f, 0.0f, 0.0f));
    }

    // Pulley 2,3,4,6,7,8 랜덤으로 생성
    Spawn_Pulley();

    // 레밸 재실행??
}

void AEnvironment::Spawn_Pulley() // 환경 리셋
{
    // 월드를 가져옵니다.
    UWorld* World = GetWorld();
    if (World && BP_Pulley)
    {
        // 스폰 파라미터를 설정합니다.
        FActorSpawnParameters SpawnParams;
        SpawnParams.Owner = this;
        SpawnParams.Instigator = GetInstigator();

        for (int32 i = 1; i < 5; i++) {
            if (i == 1) {
                AActor* SpawnedActor_top = World->SpawnActor<AActor>(BP_Pulley, FVector(-450.0f, 300.0f, 450.0f), FRotator(0.0f, 0.0f, 0.0f), SpawnParams);
                AActor* SpawnedActor_bottom = World->SpawnActor<AActor>(BP_Pulley, FVector(-450.0f, 300.0f, 0.0f), FRotator(0.0f, 0.0f, 0.0f), SpawnParams);
                if (SpawnedActor_top && SpawnedActor_bottom)
                {
                    APulley* world_puley = Cast<APulley>(SpawnedActor_top);
                    if (world_puley) {
                        world_puley->Pulley_Number = 1;
                        world_puley->AttachCableToEndEffector();
                    }

                    world_puley = Cast<APulley>(SpawnedActor_bottom);
                    if (world_puley) {
                        world_puley->Pulley_Number = 5;
                        world_puley->AttachCableToEndEffector();
                    }
                }
                continue;
            }
            /*
            else if (i == 5) {
                AActor* SpawnedActor = World->SpawnActor<AActor>(BP_Pulley, FVector(-450.0f, 300.0f, 0.0f), FRotator(0.0f, 0.0f, 0.0f), SpawnParams);
                if (SpawnedActor)
                {
                    APulley* world_puley = Cast<APulley>(SpawnedActor);
                    if (world_puley) {
                        world_puley->Pulley_Number = i;
                        world_puley->AttachCableToEndEffector();
                    }
                }
                continue;
            }
            */

            FVector Location = SpawnPulleyOnCircle(468.59f, 450.0f, i);
            AActor* SpawnedActor_top = World->SpawnActor<AActor>(BP_Pulley, Location, FRotator(0.0f, 0.0f, 0.0f), SpawnParams);
            Location.Z = 450.0f;
            AActor* SpawnedActor_bottom = World->SpawnActor<AActor>(BP_Pulley, Location, FRotator(0.0f, 0.0f, 0.0f), SpawnParams);
            if (SpawnedActor_top && SpawnedActor_bottom)
            {
                APulley* world_puley = Cast<APulley>(SpawnedActor_top);
                if (world_puley) {
                    world_puley->Pulley_Number = i;
                    world_puley->AttachCableToEndEffector();
                }
                world_puley = Cast<APulley>(SpawnedActor_bottom);
                if (world_puley) {
                    world_puley->Pulley_Number = i+4;
                    world_puley->AttachCableToEndEffector();
                }
            }

        }
    }
}

FVector AEnvironment::SpawnPulleyOnCircle(float Radius, float Height, int32 NumPulley)
{
    float Angle = 0.0f;
    float Radians = 0.0f;
    FVector Location;

    switch (NumPulley) {
    case 2:
        Angle = FMath::RandRange(0.0f, 90.0f);
        Radians = FMath::DegreesToRadians(Angle);
        Location.X = Radius * FMath::Cos(Radians);
        Location.Y = Radius * FMath::Sin(Radians);
        Location.Z = 0.0f;
        return Location;
    case 3:
        Angle = FMath::RandRange(270.0f, 360.0f);
        Radians = FMath::DegreesToRadians(Angle);
        Location.X = Radius * FMath::Cos(Radians);
        Location.Y = Radius * FMath::Sin(Radians);
        Location.Z = 0.0f;
        return Location;
    case 4:
        Angle = FMath::RandRange(180.0f, 270.0f);
        Radians = FMath::DegreesToRadians(Angle);
        Location.X = Radius * FMath::Cos(Radians);
        Location.Y = Radius * FMath::Sin(Radians);
        Location.Z = 0.0f;
        return Location;
        /*
        
    case 6:
        Angle = FMath::RandRange(0.0f, 90.0f);
        Radians = FMath::DegreesToRadians(Angle);
        Location.X = Radius * FMath::Cos(Radians);
        Location.Y = Radius * FMath::Sin(Radians);
        Location.Z = Height;
        return Location;
    case 7:
        Angle = FMath::RandRange(270.0f, 360.0f);
        Radians = FMath::DegreesToRadians(Angle);
        Location.X = Radius * FMath::Cos(Radians);
        Location.Y = Radius * FMath::Sin(Radians);
        Location.Z = Height;
        return Location;
    case 8:
        Angle = FMath::RandRange(180.0f, 270.0f);
        Radians = FMath::DegreesToRadians(Angle);S
        Location.X = Radius * FMath::Cos(Radians);
        Location.Y = Radius * FMath::Sin(Radians);
        Location.Z = Height;
        return Location;
        */
    default:
        // 기본 위치 반환
        return FVector(0.0f, 0.0f, 0.0f);
    }
}

float AEnvironment::GetReward(const FVector& CurrentPosition, const FRotator& CurrentRotation, float TimeStep)
{
    // 현재 타임스텝에 해당하는 목표 지점을 선택
    int32 TargetIndex = FMath::Min(TimeStep, TargetPositions.Num() - 1);
    FVector TargetPosition = TargetPositions[TargetIndex];

    // 초기 보상 값 설정
    Reward = 0.0f;

    // 목표 지점에 도달했는지 확인
    if (FVector::Dist(CurrentPosition, TargetPosition) < 10.0f)
    {
        // 목표 지점에 도달하면 큰 보상 부여, 시간에 따라 보상이 감소
        return 100.0f - TimeStep;
    }

    // 경로 유지 보상 계산
    float MinDistanceToPath = MAX_FLT;
    for (const FVector& TargetPos : TargetPositions)
    {
        float Distance = FVector::Dist(CurrentPosition, TargetPos);
        if (Distance < MinDistanceToPath)
        {
            MinDistanceToPath = Distance;
        }
    }

    // 경로를 유지하고 있으면 작은 보상 부여
    if (MinDistanceToPath < MaxDeviation)
    {
        Reward += 1.0f;
    }
    else
    {
        // 경로에서 벗어나면 벌점 부여
        Reward -= 50.0f;
    }

    // 엔드이펙터의 회전 확인
    if (!CurrentRotation.Equals(FRotator(0.0f, 0.0f, 0.0f), 5.0f)) // 허용 오차 5.0도 이내로 설정
    {
        // 회전하면 벌점 부여
        Reward -= 100.0f;
    }

    // 시간 경과에 따른 벌점 부여
    Reward -= 1.0f;

    return Reward;
}


// 행동 설정
void AEnvironment::step(TArray<float> input_CableLength, TArray<float> input_Tension, float DeltaTime) // 8개의 케이블 길이, 8개의 텐션
{
    // 케이블 1부터 8까지 힘과 텐션 입력
    // 상태는 엔드이펙터의 위치, 그리고 돌아간 정도
    TArray<FVector> reslut;

    


    // 
    // 모든 액터들을 담을 배열
    UWorld* World = GetWorld();
    if (World && BP_EndEffector)
    {
        TArray<AActor*> AllActors;
        // 월드에서 모든 액터 가져오기
        UGameplayStatics::GetAllActorsOfClass(World, AActor::StaticClass(), AllActors);

        // "Pulley"로 시작하는 이름을 가진 액터들을 담을 배열
        TArray<AActor*> world_Pulley_Actors;

        // 모든 액터를 순회하면서 필터링
        for (AActor* Actor : AllActors)
        {
            if (Actor && Actor->GetName().StartsWith(TEXT("Pulley")))
            {
                world_Pulley_Actors.Add(Actor);
            }
        }

        // 추론한대로 풀리한테 명령 전달
        for (int32 i = 0; i < 8; i++) {
            APulley* WPulleyActor = Cast<APulley>(world_Pulley_Actors[i]);
            WPulleyActor->CableLength = input_CableLength[i];
            WPulleyActor->Tension = input_Tension[i];
            if (WPulleyActor->Tension > WPulleyActor->MAX_Tension || WPulleyActor->Tension < WPulleyActor->MIN_Tension || WPulleyActor->CableLength > WPulleyActor->MaxCableLength || WPulleyActor->CableLength < WPulleyActor->MinCableLength) {
                Reward = 0; // 실패가 되었으므로...
                done = true;
            }
        }

        TArray<AActor*> Put_EndEffector;
        UGameplayStatics::GetAllActorsOfClass(GetWorld(), AEndEffector::StaticClass(), Put_EndEffector);

        // Assuming we have found at least one instance of AMyActorB
        if (Put_EndEffector.Num() > 0)
        {
            AEndEffector* WEnd_Effector = Cast<AEndEffector>(Put_EndEffector[0]);
            Reward = GetReward(WEnd_Effector->GetActorLocation(), WEnd_Effector->GetActorRotation(), DeltaTime);
        }
    }
    done = false; // 아직 실험 안끝남
}

