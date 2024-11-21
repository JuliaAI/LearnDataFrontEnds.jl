# # TWO PARTIAL LEARNER IMPLEMENTATIONS

using LearnAPI

struct Names{L}
    learner::L
    names
end

LearnAPI.learner(model::Names) = model.learner

# this learner does not store feature names:
struct LearnerNotReportingNames end
LearnAPI.fit(learner::LearnerNotReportingNames, data) = Names(learner, nothing)
@trait(
    LearnerNotReportingNames,
    functions = [:(LearnerAPI.learner), :(Learner.fit)],
)

# this learner does store feature names:
struct LearnerReportingNames end
LearnAPI.fit(learner::LearnerReportingNames, data) = Names(learner, collect(keys(data)))
LearnAPI.feature_names(model::Names) = model.names
@trait(
    LearnerReportingNames,
    functions = [:(LearnAPI.feature_names), :(LearnerAPI.learner), :(Learner.fit)],
)

