# %%
from pylab import np, plt, mpl
# You should be using the expit function from scipy.special.expit to compute the sigmoid function. This would give you better precision.
# log10(1000) - log10(100), log10(100) - log10(10)

# 字体
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '900'
plt.rcParams['font.size'] = 12  # 13 is too big, commenting out is too small

# 颜色设置壹/叁
plt.rcParams['axes.facecolor'] = '#272822'   # '#3d444c'
plt.rcParams['axes.labelcolor'] = '#d5d1c7'  # tint grey
plt.rcParams['axes.axisbelow'] = 'False'   # 'line'
plt.rcParams['ytick.color'] = '#d5d1c7'  # 'white'
plt.rcParams['xtick.color'] = '#d5d1c7'  # 'white'
plt.rcParams['text.color'] = '#d5d1c7'  # 'white'
plt.rcParams['grid.color'] = '#d5d1c7'  # grid color
plt.rcParams['grid.linestyle'] = '--'      # solid
plt.rcParams['grid.linewidth'] = 0.3       # in points
plt.rcParams['grid.alpha'] = 0.8       # transparency, between 0.0 and 1.0


def get_fig():
    # return plt.figure(figsize=(12, 6), facecolor='#272822')  # '#3d444c')
    return plt.figure(figsize=(24, 12), facecolor='#272822')  # '#3d444c')
    # return plt.figure(figsize=(12, 6))


def sm_sigmoid(x, a):
    return 2.0 / (1.0 + np.exp(-a*x)) - 1.0

def inv_sigmoid(x, a1=2.10871359, a2=7.36725304, a3=62.98712805):
    return a1 * x + a2 / (1+np.exp(-a3 * x)) - a2/2

# %% Sliding mode observer

get_fig()
x = np.arange(-2, 2, 0.001)
for a in [0.05494, 0.2, 1, 5, 20, 200]:
    y = sm_sigmoid(x, a)
    plt.plot(x, y, label=f'$a={a}$')
plt.legend()
plt.grid()

# %% Inverter characteristics

# tag_selected=='Mini6PhaseIPMInverter-528-0927-phaseB-80V[294-Points]-1A-[ADC offset-compensated]':
lut_current = [0.7277814358743883, 0.7227503253769825, 0.7175834047757551, 0.7124987132866172, 0.7073820178286824, 0.7022768335804146, 0.697228121116677, 0.6920964044481134, 0.6869885363871194, 0.6818877394235473, 0.6768253472225314, 0.6716576523945931, 0.6665607274306012, 0.6614540458244105, 0.6563790070544231, 0.6512440393264938, 0.6461346739075765, 0.6410732109518189, 0.6359892936893919, 0.6308537569827724, 0.625776189418606, 0.6206553647515459, 0.6155841971151135, 0.6104754517703854, 0.6053660863514682, 0.6002572881791906, 0.5951445166823602, 0.5900679294589509, 0.5849448843027556, 0.5798371192987846, 0.5747509824132477, 0.5696561740153427, 0.5645095911546979, 0.5594312484977951, 0.554310217716689, 0.5492400312870389, 0.544130149716981, 0.539022435808509, 0.5339271623549622, 0.5288277077307659, 0.5237147292538642, 0.5186450589753547, 0.5135392555189233, 0.5083901429980741, 0.5032868172403229, 0.49817352872632653, 0.49313777286865484, 0.4880114246915694, 0.482902884594862, 0.4777730783783393, 0.4726874576439431, 0.4676200069885437, 0.4624828187714791, 0.45739549543113905, 0.45231002971529016, 0.4471997350511147, 0.44208180031082706, 0.4369841530816485, 0.4318456744867322, 0.42679943885564936, 0.4216859961891312, 0.4165811730734568, 0.41151077966373534, 0.40637630210618464, 0.40128448669207517, 0.3961735477049993, 0.39107453128965736, 0.38596490779516984, 0.38083388827705644, 0.3757503070324861, 0.37065508554046356, 0.3655545986139859, 0.36045227917775396, 0.355351611251967, 0.3502484405127631, 0.34513432581053155, 0.34004085888597707, 0.33495676149026604, 0.3298722510004375, 0.32473325625438065, 0.3196265486674277, 0.3145289265529858, 0.30944168028890673, 0.30434170951356976, 0.29922194312955314, 0.29411582963602717, 0.28901289099163147, 0.2839258768223606, 0.2788059554197967, 0.2736935961510586, 0.2686033551711332, 0.2635122369074737, 0.2584320366260798, 0.2533173546772088, 0.24817807587481952, 0.24313924822504054, 0.23800095062943374, 0.2328962045900204, 0.22778152264114937, 0.2226891919419246, 0.21762047602341025, 0.2125140013972909, 0.20742432853003037, 0.20228734729303735, 0.19717382668423278, 0.19206379009562766, 0.18698913843899578, 0.1818648037709742, 0.17677088564718432, 0.1716628617016176, 0.16658432592104971, 0.16143859522940224, 0.15634339885211637, 0.1512689801563181, 0.14618969612980137, 0.14107896239074622, 0.13597340401950406, 0.1308644127235613, 0.1257649710897461, 0.12063673930740747, 0.11554913104470958, 0.11044410181498913, 0.10536285624093285, 0.10024955473655545, 0.0951444995260729, 0.09009039259009499, 0.08491301213404079, 0.07982534671366624, 0.07474468137663047, 0.06958035452148752, 0.06452407905789853, 0.05941579790278688, 0.05432379586020289, 0.049280154581422084, 0.04410592974873419, 0.03908211014599776, 0.03387984323753025, 0.028882778537015794, 0.02372104880545625, 0.018729326615657727, 0.01359873933082335, 0.008707116427513643, 0.0036892265873194607, -0.001523180266190801, -0.006760591697979452, -0.011901585144065703, -0.016970535937329813, -0.027229638090886805, -0.032422048599689016, -0.03753284642462406, -0.04262113018713692, -0.04776660314962425, -0.05278083446550014, -0.0579059759227879, -0.06303177555938254, -0.06813651129309363, -0.07319903556298361, -0.07830029671617218, -0.08345073806640102, -0.088592868690445, -0.09361286842833041, -0.09870018596630027, -0.10384078736268623, -0.10894886855253214, -0.11403959277463287, -0.11914011261007579, -0.12423234025227747, -0.12933234913273217, -0.1344520904020121, -0.13954582839251797, -0.1446038324428129, -0.1497540023806802, -0.1548450816731965, -0.15995839616795493, -0.16507898874020677, -0.1701486590187161, -0.17527785901745616, -0.18036029191234101, -0.1854676698029565, -0.1905705053903292, -0.19568083902564778, -0.2007704859121462, -0.20589611095801943, -0.21097166587914745, -0.21605764774613706, -0.22117325894026635, -0.2263011815515358, -0.23140991477190823, -0.2365056143099537, -0.24161717337121869, -0.24666885370401517, -0.2518334005296107, -0.2568892880138188, -0.26197421159776496, -0.2671296729508494, -0.27219567820984986, -0.27730723813714026, -0.2824310566540212, -0.2875071788217887, -0.29258072109987837, -0.29770237195517363, -0.3028196346597479, -0.30788009129398636, -0.3129932247894354, -0.3181405558960291, -0.32320219898507074, -0.32833967125846775, -0.3334499659226432, -0.33856413172037353, -0.34364260341506153, -0.3487279141123632, -0.35383023354859516, -0.35893392043893974, -0.36406372665546244, -0.3691154598158086, -0.3742133911013196, -0.3792847908325603, -0.38447873402446187, -0.38952974405359586, -0.3946369929064262, -0.39972718192482737, -0.40486780306659254, -0.40996297259709086, -0.41506498286225374, -0.42015837184452187, -0.4252757124913825, -0.430386110212581, -0.4354754470619849, -0.4405997566152698, -0.4456653747609147, -0.45080648520703304, -0.45589011927915296, -0.46096035680430175, -0.4660481971618079, -0.4711633952618196, -0.4762866976275599, -0.4814016385180266, -0.4864573709836874, -0.4916003407863476, -0.4967043377137868, -0.5017725106343729, -0.5069030399821077, -0.5119774075824072, -0.5170646806932739, -0.5221721356601503, -0.5273123177270358, -0.5323969061591506, -0.5375160273542414, -0.5425891045766893, -0.5477010507513097, -0.5528476336119544, -0.5579072900387199, -0.5630065376828447, -0.568119723139818, -0.5732219646337636, -0.5782949907607126, -0.5834418827924265, -0.5885450535316304, -0.5936289707940574, -0.5987319354192152, -0.6038027419230543, -0.6089438523691727, -0.6139898299246852, -0.6191748686433851, -0.624223530011624, -0.6293260304471653, -0.6344522236057035, -0.6395503869860227, -0.6446725539924586, -0.6497361335143031, -0.654869191656217, -0.65993839774511, -0.6650708886403844, -0.6701627031884685, -0.6752566862641637, -0.6803470554158487, -0.6855063887685136, -0.6905433839085381, -0.6956943540538785, -0.7007500354240405, -0.7058628597484203, -0.7109532808616297, -0.7160770993785107, -0.7211706693600882, -0.7262920621398132, -0.7313582224173609, -0.7365045982979342, -0.7415676097071139, -0.7466609216131211, -0.7517617696721921, -0.7569143921940029, -0.7620071879488695, -0.7671112879333316, -0.7722029474628684]
lut_voltage = [5.1354619426464385, 5.125713778836106, 5.1182636913770025, 5.109716800064513, 5.102783314079275, 5.095365442765192, 5.086107033641206, 5.077569642627398, 5.07053209502964, 5.062280873450257, 5.054689147536364, 5.046646473534467, 5.039169539287846, 5.029901222833241, 5.0200407268677845, 5.012935049051511, 5.005590261621253, 4.996713007597982, 4.987225829203335, 4.979772026495248, 4.9704591010720955, 4.964863174002494, 4.95585791276429, 4.946597849531785, 4.938314013435695, 4.929525951368761, 4.920931575883381, 4.91203160601377, 4.904209041028498, 4.895036108611867, 4.886393417569212, 4.878583236747214, 4.8709543410229905, 4.860647027910705, 4.852591138361145, 4.842926797149645, 4.833837288960161, 4.825833023184921, 4.817729225110025, 4.807454959527147, 4.8000172562313175, 4.791522811417281, 4.781645800347374, 4.7729989783635425, 4.765699613965713, 4.757274962138967, 4.744992080535823, 4.739952401583072, 4.732508921897798, 4.725352858723534, 4.715241293333265, 4.704533417490967, 4.694999568987309, 4.686399824144427, 4.676991921715242, 4.667980052703206, 4.657039267988814, 4.648908224754715, 4.640003716911987, 4.628796586084661, 4.619850777490427, 4.614301104837641, 4.605280965283, 4.5946358589619685, 4.584848464200844, 4.575493008270112, 4.5655883276885545, 4.5557005779036395, 4.5465239388982805, 4.535951102397194, 4.525337380836795, 4.516346980594521, 4.50512167189397, 4.494417519960907, 4.4855671040648994, 4.476542842229336, 4.463939929598477, 4.4545679585632945, 4.4445955547951606, 4.435247122330453, 4.424214253134876, 4.414611443028315, 4.405212625205615, 4.395372783946038, 4.383994275351557, 4.373871555554088, 4.35817974681428, 4.347530102520133, 4.335228228979883, 4.3237708427916255, 4.313455692149435, 4.305952742499684, 4.290452966232912, 4.282363215090065, 4.27198446354222, 4.255712868115087, 4.243363926094141, 4.226783852418181, 4.214320517101649, 4.203656834535706, 4.195534460215898, 4.188013757045369, 4.172167501335051, 4.157868174841989, 4.142548860948857, 4.1283065189273636, 4.112945072898337, 4.09711411119665, 4.079891074280156, 4.064224458219345, 4.047255391912766, 4.029944392796011, 4.012347639937023, 3.9930813113430954, 3.973628743986083, 3.9543793461887993, 3.9332828114680383, 3.911039918920287, 3.8885133337707645, 3.8660144094726383, 3.8420003564100838, 3.817584095169249, 3.7883048934400683, 3.7602393656925126, 3.728110377468368, 3.696668121408663, 3.6620403554459635, 3.6243790323976177, 3.5838365380936748, 3.53845753439671, 3.4792313907110137, 3.4223888087441536, 3.3557703934553107, 3.2800110453885063, 3.186543244506134, 3.0871047613811804, 2.9803406906091707, 2.847968577736909, 2.6498419445570947, 2.3427054029063985, 1.8226161676117363, 1.0853042224774616, 0.4880580585776978, -0.20151035984325216, -0.759834671274598, -1.385758771920178, -2.065301886277229, -2.7467226043314628, -2.9488513660687694, -3.0721506937020946, -3.1603752390400484, -3.24238586158026, -3.3232519882481006, -3.400210954704227, -3.4569277897824575, -3.514763936714317, -3.5659023990578804, -3.610498030815599, -3.6485144935615295, -3.686380640278124, -3.7223571109034075, -3.7532150336424173, -3.7835245457944806, -3.81331870354926, -3.837637917983346, -3.8620430334772338, -3.887373583717606, -3.908365641133655, -3.9294014674736113, -3.953683105065427, -3.971404989935008, -3.9908550804593648, -4.011198805277378, -4.028834789087158, -4.045289730752526, -4.063250716575838, -4.078663362026735, -4.0925476718257965, -4.10514645351548, -4.119649780952658, -4.132626418186266, -4.144941922966372, -4.160794362098073, -4.176371370510355, -4.189599492860968, -4.202471245757925, -4.217055503269086, -4.229364815967555, -4.241482096192988, -4.257130543040828, -4.270449104424359, -4.282082406352887, -4.296458124946818, -4.30691286156359, -4.318629579059011, -4.331749924548123, -4.3439241979053795, -4.354112571088709, -4.363221902599686, -4.378408255612575, -4.387736042031656, -4.3973239824820345, -4.411499017009146, -4.422046245139042, -4.429556209594564, -4.443896421146939, -4.4507460758730915, -4.458646280035812, -4.470067734830067, -4.482771002484717, -4.493141492150209, -4.5033075738396455, -4.511816056925476, -4.524172022412447, -4.534951336690304, -4.545801682371779, -4.556232880418075, -4.563617721523258, -4.573850695014883, -4.582885695565453, -4.592227104564136, -4.601172913158371, -4.609699565457173, -4.62043180259408, -4.630825008105913, -4.64060125712009, -4.64872033188311, -4.657558354955588, -4.667810329044573, -4.676252734392095, -4.685893537033121, -4.695330347390285, -4.705996506788883, -4.715315208601479, -4.722039739977292, -4.731343987825898, -4.741499330800328, -4.7502766454920025, -4.759884409263873, -4.768536600605208, -4.777445655081306, -4.786177131048357, -4.795115500484373, -4.803025620637968, -4.813268510120466, -4.821940524783294, -4.828523409044064, -4.8377297873617895, -4.84750645206816, -4.855700264823526, -4.865740391778236, -4.873562549731568, -4.8822453031094035, -4.8913393579322575, -4.899361385888528, -4.905440875562841, -4.913350995716436, -4.921640616862222, -4.9302217767999394, -4.937302686289664, -4.948138162314955, -4.9537584593394195, -4.963127537849753, -4.971723983135847, -4.980460828460402, -4.989077097067988, -4.997927504303742, -5.004363787784159, -5.013959998777144, -5.020637860043948, -5.0311933587164495, -5.038867261780908, -5.044925282685461, -5.053200449867259, -5.06047090633711, -5.068476410528677, -5.074843732406191, -5.0843779966020435, -5.09334403554971, -5.100898600313526, -5.109117605747889, -5.117587680607063, -5.125997471437878, -5.134050061430649, -5.141191670640925, -5.148996482105419, -5.1576808895917745, -5.16464038231989, -5.174069753518835, -5.181486395076845, -5.1882666725079, -5.198829601678367, -5.205994333766924, -5.214964087963573]
N = len(lut_current)
print(len(lut_current))
print(len(lut_voltage))

get_fig()
x = np.arange(0, 1, 0.001)
popt = (1.705, 7.705938744542915, 1.2*55.84230295697151)
y = inv_sigmoid(x, *popt)
plt.plot(lut_current[:int(N/2)-4], lut_voltage[:int(N/2)-4], '.', label='ori')
plt.plot(x, y, label=f'$fitted$')
plt.legend()
plt.grid()

plt.show()

print('这都拟合的啥玩意？还不如我用python制作真正的LUT呢，参考therrin2013的函数G的查表')

# %%